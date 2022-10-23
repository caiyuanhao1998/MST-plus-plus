%% plot color pics
clear; clc;
load('simulation_results\results\ARAD_1K_0314.mat');

save_file = 'simulation_results\rgb_results\mst_s\';
mkdir(save_file);

close all;
frame = 1;

recon = cube;
intensity = 5;
for channel=1:31
    img_nb = channel;  % channel number
    row_num = 1; col_num = 1;
    lam31 = [400 410 420 430 440 450 460 470 480 490 500 510 ...
            520 530 540 550 560 570 580 590 600 610 620 630 ...
            640 650 660 670 680 690 700];
    recon(find(recon>1))=1;
    name = [save_file 'frame' num2str(frame) 'channel' num2str(channel)];
    dispCubeAshwin(recon(:,:,img_nb),intensity,lam31(img_nb), [] ,col_num,row_num,0,1,name);
end
hold on;


