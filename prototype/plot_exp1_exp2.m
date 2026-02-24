%% plot_exp1_exp2.m
% 결과 CSV:
%  - results/exp1_core_compare_summary.csv
%  - results/exp2_step_sweep_summary.csv
%  - results/timeline_predicted_diffusion_seed0.csv
%  - results/timeline_predicted_consistency_seed0.csv

clear; clc;

%% EXP1 summary
T1 = readtable('results/exp1_core_compare_summary.csv', 'TextType', 'string');

% 정렬 보기 좋게
[~, idx1] = sortrows([T1.Mode, string(T1.Method)]);
T1 = T1(idx1, :);

labels1 = string(T1.Method);

figure('Color','w','Position',[80 80 1300 750]);

metricsA = ["Availability_mean", "EffectiveQoE_mean", "DeadlineMissRatio_mean", "HO_Failure_Ratio_mean"];
titlesA  = ["Availability", "Effective QoE", "Deadline Miss Ratio", "HO Failure Ratio"];

for i = 1:numel(metricsA)
    subplot(2,2,i);
    vals = T1.(metricsA(i));
    bar(vals);
    xticks(1:numel(vals));
    xticklabels(labels1);
    xtickangle(20);
    grid on;
    title(titlesA(i));
    ylabel(strrep(metricsA(i), '_', '\_'));
    for k = 1:numel(vals)
        text(k, vals(k), sprintf('%.3f', vals(k)), 'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom', 'FontSize', 9);
    end
end
sgtitle('Experiment 1 (ATG+LEO): Core Comparison');


figure('Color','w','Position',[80 80 1300 420]);
metricsB = ["MeanLatency_ms_mean", "MeanJitter_ms_mean", "MeanInterruption_ms_mean"];
titlesB  = ["Mean Latency (ms)", "Mean Jitter (ms)", "Mean Interruption (ms)"];

for i = 1:numel(metricsB)
    subplot(1,3,i);
    vals = T1.(metricsB(i));
    bar(vals);
    xticks(1:numel(vals));
    xticklabels(labels1);
    xtickangle(20);
    grid on;
    title(titlesB(i));
    ylabel('ms');
    for k = 1:numel(vals)
        text(k, vals(k), sprintf('%.2f', vals(k)), 'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom', 'FontSize', 9);
    end
end
sgtitle('Experiment 1 (ATG+LEO): Timing-related Metrics');

%% EXP2 step sweep
T2 = readtable('results/exp2_step_sweep_summary.csv', 'TextType', 'string');

figure('Color','w','Position',[80 80 1300 470]);

subplot(1,2,1); hold on; grid on;
modes = unique(T2.Mode);
for i = 1:numel(modes)
    idx = T2.Mode == modes(i);
    sub = T2(idx, :);
    [x, order] = sort(sub.SamplingSteps);
    y = sub.EffectiveQoE_mean(order);
    plot(x, y, '-o', 'LineWidth', 1.5, 'DisplayName', modes(i));
end
xlabel('Sampling Steps');
ylabel('Effective QoE');
title('Experiment 2: Effective QoE vs Step Budget');
legend('Location','best');

subplot(1,2,2); hold on; grid on;
for i = 1:numel(modes)
    idx = T2.Mode == modes(i);
    sub = T2(idx, :);
    [x, order] = sort(sub.SamplingSteps);
    y = sub.DeadlineMissRatio_mean(order);
    plot(x, y, '-o', 'LineWidth', 1.5, 'DisplayName', modes(i));
end
xlabel('Sampling Steps');
ylabel('Deadline Miss Ratio');
title('Experiment 2: Deadline Miss Ratio vs Step Budget');
legend('Location','best');

%% Timeline comparison (seed0)
Tc = readtable('results/timeline_predicted_consistency_seed0.csv', 'TextType', 'string');
Td = readtable('results/timeline_predicted_diffusion_seed0.csv', 'TextType', 'string');

figure('Color','w','Position',[100 100 1400 800]);

subplot(3,1,1);
plot(Tc.sim_time_sec, Tc.latency_ms, 'LineWidth', 1.2); hold on;
idx_out_c = find(Tc.outage == 1);
if ~isempty(idx_out_c)
    scatter(Tc.sim_time_sec(idx_out_c), zeros(size(idx_out_c)), 14, 'filled');
end
grid on;
ylabel('Latency (ms)');
title('Predicted + Consistency (seed0)');
legend({'Latency','Outage slots'}, 'Location','best');

subplot(3,1,2);
plot(Td.sim_time_sec, Td.latency_ms, 'LineWidth', 1.2); hold on;
idx_out_d = find(Td.outage == 1);
if ~isempty(idx_out_d)
    scatter(Td.sim_time_sec(idx_out_d), zeros(size(idx_out_d)), 14, 'filled');
end
grid on;
ylabel('Latency (ms)');
title('Predicted + Diffusion (seed0)');
legend({'Latency','Outage slots'}, 'Location','best');

subplot(3,1,3); hold on; grid on;
x_dm = Td.sim_time_sec(Td.deadline_miss == 1);
x_ha = Td.sim_time_sec(Td.ho_attempt == 1);
x_hf = Td.sim_time_sec(Td.ho_failure == 1);

if ~isempty(x_dm), stem(x_dm, 3 * ones(size(x_dm)), 'filled', 'LineWidth',1.0); end
if ~isempty(x_ha), stem(x_ha, 2 * ones(size(x_ha)), 'filled', 'LineWidth',1.0); end
if ~isempty(x_hf), stem(x_hf, 1 * ones(size(x_hf)), 'filled', 'LineWidth',1.2); end

yticks([1 2 3]);
yticklabels({'HO fail', 'HO attempt', 'Deadline miss'});
ylim([0.5 3.5]);
xlabel('Time (s)');
title('Event Raster (Predicted + Diffusion, seed0)');