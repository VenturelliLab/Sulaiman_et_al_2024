clear all
close all
clc
%%
data_name = 'MS_2';

load(['fitting_output/opt_res_' data_name])

num_strain = length(strain_vec);
num_media = length(media_vec);

num_theta0 = length(theta0_vec);
num_lambda = length(lambda_vec);

theta0_matrix = zeros(num_strain,num_media,num_lambda);

count = 1;
for i = 1:num_strain
    for j = 1:num_media
        
        sol_candidates = sol_matrix{i,j};
        
        each_lambda_vec = zeros(1,num_lambda);
        for q = 1:num_lambda
            
            each_theta0_vec = zeros(1,num_theta0);
            for k = 1:num_theta0
                
                each_theta0_vec(k) = sol_candidates{q,k}.fval;
            end
            
            [each_lambda_vec(q),idx_theta0] = min(each_theta0_vec);
            theta0_matrix(i,j,q) = idx_theta0;
            
        end
        
%         figure(1)
%         subplot(num_strain,num_media,count)
%         semilogx(lambda_vec,each_lambda_vec,'linewidth',2)
%         title([strain_vec{i} ', ' media_vec{j}])
%         xlabel('\lambda'); ylabel('error')
%         grid on
        
        count = count + 1;
    end
end

%% fix a lambda
lambda = 0.02;

lambda_idx = find(lambda_vec == lambda);

% growth rate constant
r_matrix = zeros(num_strain,num_media);
% resource limiting rate constant
a_matrix = zeros(num_strain,num_media);

model = @LogisticGrowth;

count = 1;
for i = 1:num_strain
    for j = 1:num_media
        
        time_temp = time_matrix{i,j};
        exp_data_mean = data_mean{i,j};
        exp_data_std = data_std{i,j};
        
        theta_opt = sol_matrix{i,j}{lambda_idx,theta0_matrix(i,j,lambda_idx)}.theta_opt;
        r_matrix(i,j) = theta_opt(1);
        a_matrix(i,j) = theta_opt(2);
        
        [t_sim, x_sim] = ode15s(@(t,x) model(t,x,theta_opt),time_temp,exp_data_mean(1));
        
        figure(2)
        subplot(num_strain,num_media,count)
        plot(t_sim,x_sim,'linewidth',2,'color','k')
        hold on
        errorbar(time_temp, exp_data_mean, exp_data_std, 'linewidth',0.1,'color','k')
        hold on
        scatter(time_temp, exp_data_mean, 10, 'k','linewidth',2)
        xlim([min(time_temp), max(time_temp)])
        title({strain_vec{i},media_vec{j}})
        
        count = count + 1;
        
    end
end

growth_rate_matrix = r_matrix;
carrying_capacity_matrix = r_matrix./a_matrix;

figure(2)
set(gcf,'position',[0,0,800,800])

save(['fitting_output/' data_name '_matrices'],'growth_rate_matrix','carrying_capacity_matrix',...
    'media_vec','strain_vec','num_media','num_strain','data_name')
saveas(gcf,['figures/' data_name '_sim.fig'])

%%
figure()
load(['fitting_output/' data_name '_matrices'])
cmap = redblue(20);
colormap(cmap);

subplot(1,2,1)
imagesc(r_matrix)
caxis([0,1])
colorbar
set(gca,'xtick',1:num_media,'xticklabel',media_vec,'xticklabelrotation',-90,...
    'ytick',1:num_strain,'yticklabel',strain_vec,'ydir','reverse','fontsize',16)
title('growth rate')
subplot(1,2,2)
imagesc(r_matrix./a_matrix)
colorbar
caxis([0,1])
set(gca,'xtick',1:num_media,'xticklabel',media_vec,'xticklabelrotation',-90,...
    'ytick',1:num_strain,'yticklabel',strain_vec,'ydir','reverse','fontsize',16)
title('carrying capacity')
set(gcf,'position',[0,0,1000,400])

saveas(gcf,['figures/' data_name '_para.fig'])