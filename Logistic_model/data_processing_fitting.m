clear all
close all
clc
%%
data_name_vec = {'MS_2'};

for iii = 1:length(data_name_vec)
    data_name = data_name_vec{iii}

    % extract data
    load([data_name '.mat'])

    if strncmp(data_name,'DSM',3)
        time = cell2mat(data(:,2));
        media = cellstr(data(:,4));

        strain = cell(length(time),1);
        for i = 1:length(time)
            strain{i} = 'DSM';
        end

        OD = cell2mat(data(:,end));
        strain_vec = {'DSM'};

    else
        time = cell2mat(data(:,2));
        media = cellstr(data(:,4));
        strain = cellstr(data(:,end));
        OD = cell2mat(data(:,end-1));
        strain_vec = unique(strain);
    end

    media_vec = unique(media);
    time_nominal = [0:3:66];

    num_strain = length(strain_vec);
    num_media = length(media_vec);
    num_time = length(time_nominal);

    % convert data to n x m cells, where n = # of strains, m = # media
    % conditions, each cell contains temporal data

    data_mean = cell(num_strain,num_media);
    data_std = data_mean;
    time_matrix = data_mean;

    for i = 1:num_strain
        strain_temp = strain_vec{i};
        strain_idx_temp = strcmp(strain_temp,strain);

        for j = 1:num_media
            media_temp = media_vec{j};
            media_idx_temp = strcmp(media_temp,media);

            for q = 1:num_time
                time_temp = time_nominal(q);
                time_idx_temp = time == time_temp;

                idx_temp = strain_idx_temp & time_idx_temp & media_idx_temp;

                OD_temp = OD(idx_temp);
                data_mean{i,j}(q,1) = mean(OD_temp);
                data_std{i,j}(q,1) = std(OD_temp);
                time_matrix{i,j}(q,1) = time_temp;

            end
        end
    end

    %% data pre-processing

    for i = 1:num_strain
        for j = 1:num_media

            % if initial condition negative, use the second measurement as IC
            if  data_mean{i,j}(1) < 0
                data_mean{i,j} = data_mean{i,j}(2:end);
                data_std{i,j} = data_std{i,j}(2:end);
                time_matrix{i,j} = time_matrix{i,j}(2:end) - time_matrix{i,j}(1);
            end

            % remove initial data points in lag phase (defined as data that does not increase more than 20% from initial)
            OD_ini = data_mean{i,j}(1);
            idx_no_increase = find(data_mean{i,j} < OD_ini * 1.2);

            data_mean{i,j}(idx_no_increase) = [];
            data_std{i,j}(idx_no_increase) = [];
            time_matrix{i,j}(idx_no_increase) = [];
            time_matrix{i,j} = time_matrix{i,j} - time_matrix{i,j}(1);

            % remove data points where OD starts to decline by more than 10% of the max
            [OD_max_temp,idx_max_temp] = max(data_mean{i,j});

            idx_OD_small = find(data_mean{i,j} < OD_max_temp * 0.9);
            idx_OD_rmv = idx_OD_small(find(idx_OD_small > idx_max_temp));

            if ismember(idx_max_temp + 1, idx_OD_rmv)
                idx_OD_rmv(idx_OD_rmv == idx_max_temp + 1) = [];
            end

            data_mean{i,j}(idx_OD_rmv) = [];
            data_std{i,j}(idx_OD_rmv) = [];
            time_matrix{i,j}(idx_OD_rmv) = [];

        end
    end
    %% parameter optimization

    % set maximum function evaluation
    fmcOpts = optimoptions('fmincon','MaxFunctionEvaluations', 2000, 'Display','off');

    % growth model
    model = @LogisticGrowth;

    % parameter lower bound
    LB = [0 0];

    % regularization parameter grid
    lambda_vec = [0.01,0.02,0.05,0.1]; 
    num_lambda = length(lambda_vec);

    num_samples = 10;
    % initial parameter guess
    theta0_vec = lhsdesign(num_samples,2)*2;

    sol_matrix = cell(num_strain,num_media);
    % estimate one parameter for each strain and medium
    for i = 1:num_strain
        for j = 1:num_media
            exp_data_temp = data_mean{i,j};
            time_temp = time_matrix{i,j};

            % create a solution NxM matrix for each condition (i.e., strain media combination), 
            % N=number of regularization parameters, M = number of initial conditions
            sol_matrix_temp = cell(num_lambda, num_samples);
            for q = 1:num_lambda
                lambda = lambda_vec(q);

                for l = 1:num_samples
                    disp(['strain = ' num2str(i) ', medium = ' num2str(j) ', lambda = ' num2str(q) ', ini = ' num2str(l)])

                    theta0 = theta0_vec(l,:);

                    objective = @(theta) CostFcn(theta,exp_data_temp,time_temp,model,lambda);

                    [theta_opt,fval] = fmincon(objective,theta0,[],[],[],[],LB,[],[],fmcOpts);

                    sol.theta_opt = theta_opt;
                    sol.fval = fval;

                    sol_matrix_temp{q,l} = sol;
                end
            end

            sol_matrix{i,j} = sol_matrix_temp;
        end
    end

    save(['fitting_output/opt_res_' data_name],'sol_matrix','data_mean',...
        'data_std','time_matrix','strain_vec','media_vec','lambda_vec','theta0_vec')
end