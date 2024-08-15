clc;
clear all;
close all;

%% Numerical Phantom creation
% Define parameters
image_size = [256, 256]; % Image size
body_center = [128, 128]; % Center of the body contour ellipse
body_axis_lengths = [80, 120]; % Major and minor axis lengths of the body contour ellipse
prostate_center = [128, 128]; % Center of the prostate
prostate_axis_lengths = [40, 25]; % Major and minor axis lengths of the prostate ellipse
rectum_center = [128, 170]; % Center of the rectum circle
rectum_radius = 15; % Radius of the rectum circle
prostate_intensity = 200; % Intensity of the prostate region
body_intensity = 150; % Intensity of the body contour
rectum_intensity = 120; % Intensity of the rectum region
background_intensity = 100; % Intensity of the background

% Create grid
[X, Y] = meshgrid(1:image_size(2), 1:image_size(1));

% Generate ellipse representing body contour
body_mask = ((Y - body_center(1)) / body_axis_lengths(1)).^2 + ...
    ((X - body_center(2)) / body_axis_lengths(2)).^2 <= 1;

% Generate ellipse representing prostate
prostate_mask = ((X - prostate_center(1)) / prostate_axis_lengths(1)).^2 + ...
    ((Y - prostate_center(2)) / prostate_axis_lengths(2)).^2 <= 1;

% Generate circle representing rectum
[x_circle, y_circle] = meshgrid(1:image_size(2), 1:image_size(1));
rectum_mask = (x_circle - rectum_center(1)).^2 + (y_circle - rectum_center(2)).^2 <= rectum_radius^2;

% Create numerical phantom
numerical_phantom = background_intensity * ones(image_size);
numerical_phantom(body_mask) = body_intensity;
numerical_phantom(prostate_mask) = prostate_intensity;
numerical_phantom(rectum_mask) = rectum_intensity;

% Displaying numerical phantom image
figure;
imshow(numerical_phantom, []);
title('Numerical Phantom of MRI of Prostate');

% Computing FFT of the numerical phantom
numerical_phantom_fft = fftshift(fft2(ifftshift(numerical_phantom)));

% Displaying FFT
figure;
imagesc(log(abs(numerical_phantom_fft) + 1));
colormap('gray');
axis square;
title('FFT of the Numerical Phantom');

%% Generating motion to numerical phantom
prostate_center2 = [125, 125]; % Center of the prostate
prostate_center3 = [115, 115];

% Generate ellipse representing prostate
prostate_mask2 = ((X - prostate_center2(1)) / prostate_axis_lengths(1)).^2 + ...
    ((Y - prostate_center2(2)) / prostate_axis_lengths(2)).^2 <= 1;
numerical_phantom2 = background_intensity * ones(image_size);
prostate_mask3 = ((X - prostate_center3(1)) / prostate_axis_lengths(1)).^2 + ...
    ((Y - prostate_center3(2)) / prostate_axis_lengths(2)).^2 <= 1;
numerical_phantom3 = background_intensity * ones(image_size);


%Creating motion numerical phantom
numerical_phantom2(body_mask) = body_intensity;
numerical_phantom2(prostate_mask2) = prostate_intensity;
numerical_phantom2(rectum_mask) = rectum_intensity;
numerical_phantom3(body_mask) = body_intensity;
numerical_phantom3(prostate_mask3) = prostate_intensity;
numerical_phantom3(rectum_mask) = rectum_intensity;

% Computing FFT of the numerical phantom
numerical_phantom_fft2 = fftshift(fft2(ifftshift(numerical_phantom2)));
numerical_phantom_fft3 = fftshift(fft2(ifftshift(numerical_phantom3)));

% Displaying numerical phantom image
figure;
% Display numerical_phantom and numerical_phantom2 on the same figure
subplot(1, 3, 1);
imshow(numerical_phantom, []);
title('Original Numerical Phantom');
subplot(1, 3, 2);
imshow(numerical_phantom2, []);
title('Moved prostate Numerical Phantom');
subplot(1, 3, 3);
imshow(numerical_phantom3, []);
title('Moved prostate Numerical Phantom 2');


%Taking half of one image and half of the other
combined_image_fft2 = zeros(size(numerical_phantom_fft));% Initialize a new image to store the combined image
combined_image_fft3 = zeros(size(numerical_phantom_fft));% Initialize a new image to store the combined image
for row = 1:size(numerical_phantom_fft, 1)
    % Copy rows from numerical_phantom for the upper half
    if row <= 125
        combined_image_fft2(row, :) = numerical_phantom_fft(row, :);
        combined_image_fft3(row, :) = numerical_phantom_fft(row, :);
    % Copy rows from numerical_phantom2 for the lower half
    else
        combined_image_fft2(row, :) = numerical_phantom_fft2(row, :);
        combined_image_fft3(row, :) = numerical_phantom_fft3(row, :);
    end
end
combined_image2 = fftshift(ifft2(ifftshift(combined_image_fft2)));
combined_image3 = fftshift(ifft2(ifftshift(combined_image_fft3)));

% Display the combined images
figure;
subplot(1, 2, 1);
imshow(abs(combined_image2), []);
title('Combined Numerical Phantom 2 (halfs)');
subplot(1, 2, 2);
imshow(abs(combined_image3), []);
title('Combined Numerical Phantom 3 (halfs)');

%% Creating additional motion numerical phantoms
% Loop increasing the displacement and plot

I = mat2gray(numerical_phantom,[0,256]);
Ge = entropy(I);
phi_x=[];
[X, Y] = meshgrid(1:image_size(2), 1:image_size(1));

for index_i=0:15
    
    prostate_centeri = [128-index_i, 128-index_i];
    prostate_maski = ((X - prostate_centeri(1)) / prostate_axis_lengths(1)).^2 + ...
    ((Y - prostate_centeri(2)) / prostate_axis_lengths(2)).^2 <= 1;
    numerical_phantom_i = background_intensity * ones(image_size);

    %Creating motion numerical phantom
    numerical_phantom_i(body_mask) = body_intensity;
    numerical_phantom_i(prostate_maski) = prostate_intensity;
    numerical_phantom_i(rectum_mask) = rectum_intensity;

    % Computing FFT of the numerical phantom
    numerical_phantom_fft_i = fftshift(fft2(ifftshift(numerical_phantom_i)));

    %Taking half of one image and half of the other
    combined_image_fft_i = zeros(size(numerical_phantom_fft));% Initialize a new image to store the combined image
    for row = 1:size(numerical_phantom_fft, 1)
        % Copy rows from numerical_phantom for the upper half
        if row <= 125
            combined_image_fft_i(row, :) = numerical_phantom_fft(row, :);
        % Copy rows from numerical_phantom2 for the lower half
        else
            combined_image_fft_i(row, :) = numerical_phantom_fft_i(row, :);
        end
    end
    combined_image_i = fftshift(ifft2(ifftshift(combined_image_fft_i)));
    
    % Gradient Entropy (normalised)
    I_i = mat2gray(abs(combined_image_i),[0,256]);
    Ge_i = entropy(I_i);
    phi_x(index_i+1) = Ge_i;

end
figure;
imshow(combined_image_i, []);

%Plotting the graph for variable displacement
pixel_displacement=0:15;
figure
plot(pixel_displacement,phi_x)
title('Entropy as a function of prostate displacement', 'FontSize', 20)
xlabel('Displacement of the prostate (in pixels)', 'FontSize', 18)
ylabel('Entropy quality metric', 'FontSize', 18)
set(gca, 'FontSize', 16);   

%% Loop for fixed x different rows

phi_fixedx=[];
for index_row=1:255
    for row = 1:size(numerical_phantom_fft, 1)
        % Copy rows from numerical_phantom for the upper half
        if row <= index_row
            combined_image_fft_fixedx(row, :) = numerical_phantom_fft(row, :);
        % Copy rows from numerical_phantom2 for the lower half
        else
            combined_image_fft_fixedx(row, :) = numerical_phantom_fft2(row, :);
        end
    end
    combined_image_fixedx = fftshift(ifft2(ifftshift(combined_image_fft_fixedx)));
    
    % Gradient Entropy, normalised?
    I_fixedx = mat2gray(abs(combined_image_fixedx),[0,256]);
    Ge_fixedx = entropy(I_fixedx);
    phi_fixedx(index_row) = Ge_fixedx;
end
%Plotting the graph for variable displacement
row_number=1:255;
figure
plot(row_number,phi_fixedx)
title('Entropy as a function of row number in which images were combined')
xlabel('Row number')
ylabel('Entropy quality metric')

%% Applying evaluation of motion metrics

% Gradient
I = mat2gray(numerical_phantom,[0,256]);
[Gx,Gy] = imgradientxy(I,'central');
f1=sum(abs(Gy),'all');% divide to normalise by mean pixel value
I2 = mat2gray(real(combined_image2),[0,256]);%Add max and min possible for that image? [amin amax]
[Gx2,Gy2] = imgradientxy(I2,'central');
f1_2=sum(abs(Gy2),'all');% divide to normalise by mean(real(combined_image2),'all') ?
I3 = mat2gray(real(combined_image3),[0,256]);
[Gx3,Gy3] = imgradientxy(I3,'central');
f1_3=sum(abs(Gy3),'all');% divide to normalise by mean pixel value
f1_grad_images=[f1; f1_2; f1_3];


%Trying gradient of K-space calculation
IK = mat2gray(real(numerical_phantom_fft),[0,256]);
[Gxk,Gyk] = imgradientxy(IK,'central');
f1_fft=sum(abs(Gyk),'all')/mean(real(numerical_phantom_fft),'all');
IK2 = mat2gray(real(combined_image_fft2),[0,256]);
[Gxk2,Gyk2] = imgradientxy(IK2,'central');
f1_fft2=sum(abs(Gyk2),'all')/mean(real(combined_image_fft2),'all');
IK3 = mat2gray(real(combined_image_fft3),[0,256]);
[Gxk3,Gyk3] = imgradientxy(IK3,'central');
f1_fft3=sum(abs(Gyk3),'all')/mean(real(combined_image_fft3),'all');
f1_grad_kspace=[f1_fft; f1_fft2; f1_fft3];

% Entropy
[h, binLocations]=imhist(I);
h(h==0) = []; %remove zero entries
h = h ./ numel(I); %normalising h
e=-sum(h.*log(h));

[h2, binLocations2]=imhist(I2);
h2(h2==0) = [];%remove zero entries
h2 = h2 ./ numel(I2);%normalising h
e2=-sum(h2.*log(h2));

[h3, binLocations3]=imhist(I3);
h3(h3==0) = [];%remove zero entries
h3 = h3 ./ numel(I3);%normalising h
e3=-sum(h3.*log(h3));
Entropy_phantom=[e; e2; e3];

% Entropy 2 (normalised)
Ge = entropy(I);
Ge2 = entropy(I2);
Ge3 = entropy(I3);
Entropy2_phantom=[Ge; Ge2; Ge3];

%Gradient Entropy
h_e=abs(Gy)./f1;
Grad_entropy=-sumabs(h_e.*log2(abs(h_e)));
h_e2=abs(Gy2)./f1_2;
Grad_entropy2=-sumabs(h_e2.*log2(abs(h_e2)));
h_e3=abs(Gy3)./f1_3;
Grad_entropy3=-sumabs(h_e3.*log2(abs(h_e3)));
Grad_Entropy_phantom=[Grad_entropy; Grad_entropy2; Grad_entropy3];

%hmax(max(h_e2.*log2(abs(h_e2))))
B = imresize(mat2gray(h_e2.*log2(abs(h_e2))),[8 8]);
figure
imshow(B,[],InitialMagnification='fit')%make it bigger in display

%% Calculate the normalised gradient squared (R13)
F13_0=sum((abs(Gy)./sum(abs(Gy),'all')).^2,'all');
F13_2=sum((abs(Gy2)./sum(abs(Gy2),'all')).^2,'all');
F13_3=sum((abs(Gy3)./sum(abs(Gy3),'all')).^2,'all');
Norm_Grad_squared=[F13_0; F13_2; F13_3];

% Laplacian 1
Laplace=[-1 -2 -1; -2 12 -2; -1 -2 -1];
k1_I=sumabs(conv2(I, Laplace, 'same'));
k1_I2=sumabs(conv2(I2, Laplace, 'same'));
k1_I3=sumabs(conv2(I3, Laplace, 'same'));
Laplacian1=[k1_I;k1_I2;k1_I3];

% Laplacian 2
Laplace2=[0 1 0; 1 -4 1; 0 1 0];
k2_I=sumabs(conv2(I, Laplace2, 'same'));
k2_I2=sumabs(conv2(I2, Laplace2, 'same'));
k2_I3=sumabs(conv2(I3, Laplace2, 'same'));
Laplacian2=[k2_I;k2_I2;k2_I3];

% Autocorrelation 1
for AC_index=1:256
    for AC_indexj=1:256
        if AC_indexj+2<256%How should I handel edge cases?
            Plus1(AC_index,AC_indexj)=I(AC_index,AC_indexj)*I(AC_index,AC_indexj+1);
            Plus1_2(AC_index,AC_indexj)=I2(AC_index,AC_indexj)*I2(AC_index,AC_indexj+1);
            Plus1_3(AC_index,AC_indexj)=I3(AC_index,AC_indexj)*I3(AC_index,AC_indexj+1);
        end
        if AC_indexj+2<256
            Plus2(AC_index,AC_indexj)=I(AC_index,AC_indexj)*I(AC_index,AC_indexj+2);
            Plus2_2(AC_index,AC_indexj)=I2(AC_index,AC_indexj)*I2(AC_index,AC_indexj+2);
            Plus2_3(AC_index,AC_indexj)=I3(AC_index,AC_indexj)*I3(AC_index,AC_indexj+2);
        end
    end    
end
Auto_correct_1=sum(I.^2,'all')-sum(Plus1,'all');
Auto_correct_1_2=sum(I2.^2,'all')-sum(Plus1_2,'all');
Auto_correct_1_3=sum(I3.^2,'all')-sum(Plus1_3,'all');
Auto_Correlation_1=[Auto_correct_1;Auto_correct_1_2;Auto_correct_1_3];

% Autocorrelation 2
Auto_correct_2=sum(Plus1,'all')-sum(Plus2,'all');
Auto_correct_2_2=sum(Plus1_2,'all')-sum(Plus2_2,'all');
Auto_correct_2_3=sum(Plus1_2,'all')-sum(Plus2_3,'all');
Auto_Correlation_2=[Auto_correct_2;Auto_correct_2_2;Auto_correct_2_3];

%Cube of Normalised intensities
CNI=sum((I./sum(I,'all'))^3,'all')
CNI_2=sum((I2./sum(I2,'all'))^3,'all')
CNI_3=sum((I3./sum(I3,'all'))^3,'all')
Cube_norm_intensities=[CNI;CNI_2;CNI_3]

%Sum of squares Entropy normalisation (David's paper)
B_max=sqrt(sum(I.^2,'all'))
Entropy_SSnorm=-sum((I./B_max)*log(I./B_max),'all')
B_max2=sqrt(sum(I2.^2,'all'))
Entropy_SSnorm2=-sum((I2./B_max2)*log(I2./B_max2),'all')
B_max3=sqrt(sum(I3.^2,'all'))
Entropy_SSnorm3=-sum((I3./B_max3)*log(I3./B_max3),'all')
Sum_of_squares_Entropy=[Entropy_SSnorm;Entropy_SSnorm2;Entropy_SSnorm3]

%Creating a table summarising the metrics results
v=[{'No motion'};{'Little Motion'};{'Bigger motion'}];
Tab_metrics = table(v, f1_grad_images, f1_grad_kspace, Entropy_phantom, Entropy2_phantom, Grad_Entropy_phantom, Norm_Grad_squared,Laplacian1,Laplacian2,Auto_Correlation_1,Auto_Correlation_2,Cube_norm_intensities,Sum_of_squares_Entropy)
writetable(Tab_metrics,'Tablesdata.xlsx','Sheet',1,'Range','D1')


%Calculating the percentages
F1=100*(f1_grad_images-f1_grad_images(1))./f1_grad_images(1);
Entropy=100*(Entropy_phantom-Entropy_phantom(1))./Entropy_phantom(1);
Entropy2=100*(Entropy2_phantom-Entropy2_phantom(1))./Entropy2_phantom(1);
Grad_Entropy=100*(Grad_Entropy_phantom-Grad_Entropy_phantom(1))./Grad_Entropy_phantom(1);
Norm_Grad_sq=abs(100*(Norm_Grad_squared-Norm_Grad_squared(1))./Norm_Grad_squared(1));
Lapl1=100*(Laplacian1-Laplacian1(1))./Laplacian1(1);
Lapl2=100*(Laplacian2-Laplacian2(1))./Laplacian2(1);
AC1=abs(100*(Auto_Correlation_1-Auto_Correlation_1(1))./Auto_Correlation_1(1));
AC2=abs(100*(Auto_Correlation_2-Auto_Correlation_2(1))./Auto_Correlation_2(1));
Cube_norm=100*(Cube_norm_intensities-Cube_norm_intensities(1))./Cube_norm_intensities(1);
SumSquaresEntropy=100*(Sum_of_squares_Entropy-Sum_of_squares_Entropy(1))./Sum_of_squares_Entropy(1);

Tab_metrics2 = table(v, F1, Entropy, Entropy2, Grad_Entropy, Norm_Grad_sq,Lapl1,Lapl2,AC1,AC2,Cube_norm,SumSquaresEntropy)
writetable(Tab_metrics2,'Tablesdata.xlsx','Sheet',2,'Range','D1')

%% Rajski cost function
Rajski_CF=1-I./Ge;
Rajski_CF2=1-I2./Ge2;
Rajski_CF3=1-I3./Ge3;

%% Multishot code (first trial)

nk = 256 ;
nshot = 16;
lpershot = nk/nshot ; % k-space lines per shot

combinedk = zeros([nk nk]) ;

% Code to simulate images with prostate in different positions
% Numerical Phantom creation
image_size = [256, 256]; % Image size
body_center = [128, 128]; % Center of the body contour ellipse
body_axis_lengths = [80, 120]; % Major and minor axis lengths of the body contour ellipse
prostate_center = [128, 128]; % Center of the prostate
prostate_axis_lengths = [40, 25]; % Major and minor axis lengths of the prostate ellipse
rectum_center = [128, 170]; % Center of the rectum circle
rectum_radius = 15; % Radius of the rectum circle
prostate_intensity = 200; % Intensity of the prostate region
body_intensity = 150; % Intensity of the body contour
rectum_intensity = 120; % Intensity of the rectum region
background_intensity = 100; % Intensity of the background

% Create grid
[X, Y] = meshgrid(1:image_size(2), 1:image_size(1));

% Generate ellipse representing body contour
body_mask = ((Y - body_center(1)) / body_axis_lengths(1)).^2 + ...
    ((X - body_center(2)) / body_axis_lengths(2)).^2 <= 1;

% Generate ellipse representing prostate
prostate_mask = ((X - prostate_center(1)) / prostate_axis_lengths(1)).^2 + ...
    ((Y - prostate_center(2)) / prostate_axis_lengths(2)).^2 <= 1;

% Generate circle representing rectum
[x_circle, y_circle] = meshgrid(1:image_size(2), 1:image_size(1));
rectum_mask = (x_circle - rectum_center(1)).^2 + (y_circle - rectum_center(2)).^2 <= rectum_radius^2;

% Create numerical phantom
numerical_phantom = background_intensity * ones(image_size);
numerical_phantom(body_mask) = body_intensity;
numerical_phantom(prostate_mask) = prostate_intensity;
numerical_phantom(rectum_mask) = rectum_intensity;

% Computing FFT of the numerical phantom
numerical_phantom_fft = fftshift(fft2(ifftshift(numerical_phantom)));

%Displaying the Numerical Phantom Simulation
figure(1)
imshow(numerical_phantom, []);
title('Numerical Phantom of MRI of Prostate');

%% Generating motion to numerical phantom
prostate_center2 = [125, 125]; % Center of the prostate
prostate_center3 = [110, 110];

% Generate ellipse representing prostate
prostate_mask2 = ((X - prostate_center2(1)) / prostate_axis_lengths(1)).^2 + ...
    ((Y - prostate_center2(2)) / prostate_axis_lengths(2)).^2 <= 1;
numerical_phantom2 = background_intensity * ones(image_size);
prostate_mask3 = ((X - prostate_center3(1)) / prostate_axis_lengths(1)).^2 + ...
    ((Y - prostate_center3(2)) / prostate_axis_lengths(2)).^2 <= 1;
numerical_phantom3 = background_intensity * ones(image_size);


%Creating motion numerical phantom
numerical_phantom2(body_mask) = body_intensity;
numerical_phantom2(prostate_mask2) = prostate_intensity;
numerical_phantom2(rectum_mask) = rectum_intensity;
numerical_phantom3(body_mask) = body_intensity;
numerical_phantom3(prostate_mask3) = prostate_intensity;
numerical_phantom3(rectum_mask) = rectum_intensity;

for ishot = 1: nshot
    shot2k{ishot} = [ishot:lpershot:nk] ; % k-space lines in each shot
end

t2shot{1} = [1:8] ; % shots in time period 1
t2shot{2} = [9:16] ; % shots in time period 2

% loop over each time period (here 2)
for itime = 1:2
    % set the object at this time period
    if itime == 1
        shifted_object1 = numerical_phantom;
        shifted_object2 = numerical_phantom;
    else
        shifted_object1 = numerical_phantom2;
        shifted_object2 = numerical_phantom3;
    end

    % compute k-space of object
    kshifted1 = fftshift(fft2(ifftshift(shifted_object1))) ;
    kshifted2 = fftshift(fft2(ifftshift(shifted_object2))) ;

    % below aims to fill combinedk with the k-space lines acquired at this
    % time period - finds the shots and the lines within the shots.
    shots = t2shot{itime} ; % the shots in this time period

    for ishot = 1:length(shots) % loop over the shots in this time period
        this_shot = shots(ishot) ;
        klines_this_shot = shot2k{this_shot} ; % the k-space lines in this shot
        combinedk1(klines_this_shot,:) = kshifted1(klines_this_shot,:) ;
        combinedk2(klines_this_shot,:) = kshifted2(klines_this_shot,:) ;
    end
end

imgfinal1 = fftshift(ifft2(ifftshift(combinedk1))) ;
imgfinal2 = fftshift(ifft2(ifftshift(combinedk2))) ;

% Displaying numerical phantom image
figure;
imshow(abs(imgfinal1), []);
title('Numerical Phantom of MRI of Prostate');

%% Calculating metrics for this
I = mat2gray(numerical_phantom,[0,256]);
I_ms_1 = mat2gray(abs(imgfinal1),[0,256]);
I_ms_2 = mat2gray(abs(imgfinal2),[0,256]);
%Entropy log2
Ge = entropy(I);
Ge_ms_1 = entropy(I_ms_1);
Ge_ms_2 = entropy(I_ms_2);
Gradient_ms=[Ge; Ge_ms_1; Ge_ms_2];

%Gradient Entropy
[Gx,Gy] = imgradientxy(I,'central');
f1=sum(abs(Gy),'all');
h_e=abs(Gy)./f1;
Grad_entropy=-sumabs(h_e.*log2(abs(h_e)));

[Gx_ms_1,Gy_ms_1] = imgradientxy(I_ms_1,'central');
f1_ms_1=sum(abs(Gy_ms_1),'all');
h_e_ms_1=abs(Gy_ms_1)./f1_ms_1;
Grad_entropy_ms_1=-sumabs(h_e_ms_1.*log2(abs(h_e_ms_1)));

[Gx_ms_2,Gy_ms_2] = imgradientxy(I_ms_2,'central');
f1_ms_2=sum(abs(Gy_ms_2),'all');
h_e_ms_2=abs(Gy_ms_2)./f1_ms_2;
Grad_entropy_ms_2=-sumabs(h_e_ms_2.*log2(abs(h_e_ms_2)));

Grad_Entropy_phantom_ms=[Grad_entropy; Grad_entropy_ms_1; Grad_entropy_ms_2];

%Creating a table summarising the metrics results
v=[{'No motion'};{'Little Motion'};{'Bigger motion'}];
Tab_metrics = table(v, Gradient_ms, Grad_Entropy_phantom_ms)

%% Creating graphs multishot
for ishot = 1: nshot
    shot2k{ishot} = [ishot:lpershot:nk] ; % k-space lines in each shot
end

t2shot{1} = [1:8] ; % shots in time period 1
t2shot{2} = [9:16] ; % shots in time period 2

[X, Y] = meshgrid(1:image_size(2), 1:image_size(1));
phi_x=[];
% loop over each time period (here 2)
for index_i=1:85

    prostate_centeri = [128-index_i, 128-index_i];
    prostate_maski = ((X - prostate_centeri(1)) / prostate_axis_lengths(1)).^2 + ...
    ((Y - prostate_centeri(2)) / prostate_axis_lengths(2)).^2 <= 1;
    numerical_phantom_i = background_intensity * ones(image_size);
    %Creating motion numerical phantom
    numerical_phantom_i(body_mask) = body_intensity;
    numerical_phantom_i(prostate_maski) = prostate_intensity;
    numerical_phantom_i(rectum_mask) = rectum_intensity;
    % Computing FFT of the numerical phantom
    numerical_phantom_fft_i = fftshift(fft2(ifftshift(numerical_phantom_i)));

    for itime = 1:2
        % set the object at this time period
        if itime == 1
            shifted_objecti = numerical_phantom;
        else
            shifted_objecti = numerical_phantom_i;
        end
    
        % compute k-space of object
        kshiftedi = fftshift(fft2(ifftshift(shifted_objecti))) ;
    
        % below aims to fill combinedk with the k-space lines acquired at this
        % time period - finds the shots and the lines within the shots.
        shots = t2shot{itime} ; % the shots in this time period
    
        for ishot = 1:length(shots) % loop over the shots in this time period
            this_shot = shots(ishot) ;
            klines_this_shot = shot2k{this_shot} ; % the k-space lines in this shot
            combinedki(klines_this_shot,:) = kshiftedi(klines_this_shot,:) ;
        end
    end
    imgfinali = fftshift(ifft2(ifftshift(combinedki))) ;
    % Gradient Entropy (normalised)
    I_i = mat2gray(abs(imgfinali),[0,256]);
    Ge_i = entropy(I_i);
    phi_x(index_i) = Ge_i;

end
figure;
imshow(abs(imgfinali), []);

%Plotting the graph for variable displacement
pixel_displacement=1:85;
figure
plot(pixel_displacement,phi_x)
title('Entropy as a function of prostate displacement "Multi-shot"', 'FontSize', 20)
xlabel('Displacement of the prostate (in pixels)', 'FontSize', 18)
ylabel('Entropy quality metric', 'FontSize', 18)
set(gca, 'FontSize', 16);



%% Multishot Realistic code


nk = 256 ;
nshot = 16;
lpershot = nk/nshot ; % k-space lines per shot

combinedk = zeros([nk nk]) ;
% Numerical Phantom creation
% Define parameters
image_size = [256, 256]; % Image size
body_center = [128, 128]; % Center of the body contour ellipse
body_axis_lengths = [80, 120]; % Major and minor axis lengths of the body contour ellipse
prostate_center = [128, 128]; % Center of the prostate
prostate_axis_lengths = [40, 25]; % Major and minor axis lengths of the prostate ellipse
rectum_center = [128, 180]; % Center of the rectum circle
rectum_radius = 25; % Radius of the rectum circle
prostate_intensity = 100; % Intensity of the prostate region
body_intensity = 50; % Intensity of the body contour
rectum_intensity = 20; % Intensity of the rectum region
background_intensity = 5; % Intensity of the background
bladder_intensity = 120; % Intensity of the bladder
bone_intensity = 80; % Intensity of the bones
texture_intensity = 90; % Intensity of the texture lines

% Create grid
[X, Y] = meshgrid(1:image_size(2), 1:image_size(1));

% Generate ellipse representing body contour
body_mask = ((Y - body_center(1)) / body_axis_lengths(1)).^2 + ...
    ((X - body_center(2)) / body_axis_lengths(2)).^2 <= 1;

% Generate ellipse representing prostate
prostate_mask = ((X - prostate_center(1)) / prostate_axis_lengths(1)).^2 + ...
    ((Y - prostate_center(2)) / prostate_axis_lengths(2)).^2 <= 1;

% Generate circle representing rectum
[x_circle, y_circle] = meshgrid(1:image_size(2), 1:image_size(1));
rectum_mask = (x_circle - rectum_center(1)).^2 + (y_circle - rectum_center(2)).^2 <= rectum_radius^2;

% Create numerical phantom
numerical_phantom = background_intensity * ones(image_size);
numerical_phantom(body_mask) = body_intensity;
numerical_phantom(prostate_mask) = prostate_intensity;
numerical_phantom(rectum_mask) = rectum_intensity;

% Define bladder ellipse
bladder_center = [130, 105]; % Center of the bladder ellipse
bladder_axis_lengths = [37, 20]; % Major and minor axis lengths of the bladder ellipse
bladder_mask = ((X - bladder_center(1)) / bladder_axis_lengths(1)).^2 + ...
    ((Y - bladder_center(2)) / bladder_axis_lengths(2)).^2 <= 1;

% Exclude prostate region from bladder
bladder_mask(prostate_mask) = 0;

% Add bladder to numerical phantom
numerical_phantom(bladder_mask) = bladder_intensity;

% Simulate bones as semicircles in the middle part of the image
bone_center_left = [50, 120];
bone_center_right = [208, 120];
bone_radius = 25;
bone_mask_left = (X - bone_center_left(1)).^2 + (Y - bone_center_left(2)).^2 <= bone_radius^2;
bone_mask_right = (X - bone_center_right(1)).^2 + (Y - bone_center_right(2)).^2 <= bone_radius^2;
numerical_phantom(bone_mask_left) = bone_intensity;
numerical_phantom(bone_mask_right) = bone_intensity;

% Add texture lines for the prostate
texture_lines = ones(image_size) * 100;
for i = 1:10:image_size(1)
    texture_lines(i, :) = texture_intensity;
end
for j = 1:10:image_size(2)
    texture_lines(:, j) = texture_intensity;
end
text_lines=texture_lines;
texture_lines(~prostate_mask) = 100; % Limit to prostate area
numerical_phantom = numerical_phantom + texture_lines;

figure(1);
subplot(1,2,1)
imshow(numerical_phantom, []);
title('Numerical Phantom of MRI');
numerical_phantom_initial=numerical_phantom;

N=3;
realchannel = normrnd(0,N,256,256) + numerical_phantom; % N is the gaussian noise level
imaginarychannel = normrnd(0,N,256,256);
numerical_phantom = sqrt(realchannel.^2 + imaginarychannel.^2);

% Displaying numerical phantom image
figure(1);
subplot(1,2,2)
imshow(numerical_phantom, []);
title('Numerical Phantom of MRI with Rician noise');

% Computing FFT of the numerical phantom
numerical_phantom_fft = fftshift(fft2(ifftshift(numerical_phantom)));

%% Generating motion to numerical phantom
prostate_center2 = [125, 125]; % Center of the prostate
prostate_center3 = [115, 115];
%Add vertical motion
prostate_center4 = [128, 120];

% Generate ellipse representing prostate
prostate_mask2 = ((X - prostate_center2(1)) / prostate_axis_lengths(1)).^2 + ...
    ((Y - prostate_center2(2)) / prostate_axis_lengths(2)).^2 <= 1;
numerical_phantom2 = background_intensity * ones(image_size);
prostate_mask3 = ((X - prostate_center3(1)) / prostate_axis_lengths(1)).^2 + ...
    ((Y - prostate_center3(2)) / prostate_axis_lengths(2)).^2 <= 1;
numerical_phantom3 = background_intensity * ones(image_size);
prostate_mask4 = ((X - prostate_center4(1)) / prostate_axis_lengths(1)).^2 + ...
    ((Y - prostate_center4(2)) / prostate_axis_lengths(2)).^2 <= 1;
numerical_phantom4 = background_intensity * ones(image_size);

%Creating motion numerical phantom
numerical_phantom2(body_mask) = body_intensity;
numerical_phantom2(prostate_mask2) = prostate_intensity;
numerical_phantom2(rectum_mask) = rectum_intensity;
numerical_phantom2(bladder_mask) = bladder_intensity;
numerical_phantom2(bone_mask_left) = bone_intensity;
numerical_phantom2(bone_mask_right) = bone_intensity;
texture_lines2=text_lines;
texture_lines2(~prostate_mask2) = 100; % Limit to prostate area
numerical_phantom2 = numerical_phantom2 + texture_lines2;

numerical_phantom3(body_mask) = body_intensity;
numerical_phantom3(prostate_mask3) = prostate_intensity;
numerical_phantom3(rectum_mask) = rectum_intensity;
numerical_phantom3(bladder_mask) = bladder_intensity;
numerical_phantom3(bone_mask_left) = bone_intensity;
numerical_phantom3(bone_mask_right) = bone_intensity;
texture_lines3=text_lines;
texture_lines3(~prostate_mask3) = 100; % Limit to prostate area
numerical_phantom3 = numerical_phantom3+ texture_lines3;

numerical_phantom4(body_mask) = body_intensity;
numerical_phantom4(prostate_mask4) = prostate_intensity;
numerical_phantom4(rectum_mask) = rectum_intensity;
numerical_phantom4(bladder_mask) = bladder_intensity;
numerical_phantom4(bone_mask_left) = bone_intensity;
numerical_phantom4(bone_mask_right) = bone_intensity;
texture_lines4=text_lines;
texture_lines4(~prostate_mask4) = 100; % Limit to prostate area
numerical_phantom4 = numerical_phantom4+ texture_lines4;

figure;
imshow(numerical_phantom4,[])
title('Motion Generated upwards')


for ishot = 1: nshot
    shot2k{ishot} = [ishot:lpershot:nk] ; % k-space lines in each shot
end

t2shot{1} = [1:8] ; % shots in time period 1
t2shot{2} = [9:16] ; % shots in time period 2

% loop over each time period (here 2)
for itime = 1:2
    % set the object at this time period
    if itime == 1
        shifted_object1 = numerical_phantom_initial;
        shifted_object2 = numerical_phantom_initial;
        shifted_object3 = numerical_phantom_initial;
    else
        shifted_object1 = numerical_phantom2;
        shifted_object2 = numerical_phantom3;
        shifted_object3 = numerical_phantom4;
    end

    % compute k-space of object
    kshifted1 = fftshift(fft2(ifftshift(shifted_object1))) ;
    kshifted2 = fftshift(fft2(ifftshift(shifted_object2))) ;
    kshifted3 = fftshift(fft2(ifftshift(shifted_object3))) ;

    % below aims to fill combinedk with the k-space lines acquired at this
    % time period - finds the shots and the lines within the shots.
    shots = t2shot{itime} ; % the shots in this time period

    for ishot = 1:length(shots) % loop over the shots in this time period
        this_shot = shots(ishot) ;
        klines_this_shot = shot2k{this_shot} ; % the k-space lines in this shot
        combinedk1(klines_this_shot,:) = kshifted1(klines_this_shot,:) ;
        combinedk2(klines_this_shot,:) = kshifted2(klines_this_shot,:) ;
        combinedk3(klines_this_shot,:) = kshifted3(klines_this_shot,:) ;
    end
end

imgfinal1 = fftshift(ifft2(ifftshift(combinedk1))) ;
imgfinal2 = fftshift(ifft2(ifftshift(combinedk2))) ;
imgfinal3 = fftshift(ifft2(ifftshift(combinedk3))) ;

%adding noise
realchannel = normrnd(0,N,256,256) + imgfinal1;
imaginarychannel = normrnd(0,N,256,256);
imgfinal1 = sqrt(realchannel.^2 + imaginarychannel.^2);
realchannel = normrnd(0,N,256,256) + imgfinal2; 
imaginarychannel = normrnd(0,N,256,256);
imgfinal2 = sqrt(realchannel.^2 + imaginarychannel.^2);
realchannel = normrnd(0,N,256,256) + imgfinal3; 
imaginarychannel = normrnd(0,N,256,256);
imgfinal3 = sqrt(realchannel.^2 + imaginarychannel.^2);

% Displaying numerical phantom image
figure;
imshow(abs(imgfinal1), []);
title('Motion of prostate simulation');

%% Calculating metrics for this
I = mat2gray(real(numerical_phantom),[0,256]);
I_ms_1 = mat2gray(real(imgfinal1),[0,256]);
I_ms_2 = mat2gray(real(imgfinal2),[0,256]);

figure(100);
subplot(1,3,1)
imshow(abs(I), []);
title('No motion');
subplot(1,3,2)
imshow(abs(I_ms_1), []);
title('Small motion (3pixels)');
subplot(1,3,3)
imshow(abs(I_ms_2), []);
title('Bigger motion (10pixels)');


%Entropy log2
Ge = entropy(I);
Ge_ms_1 = entropy(I_ms_1);
Ge_ms_2 = entropy(I_ms_2);
Entropy2_ms=[Ge; Ge_ms_1; Ge_ms_2];

%Gradient Entropy
[Gx,Gy] = imgradientxy(I,'central');
f1=sum(abs(Gy),'all');
h_e=abs(Gy)./f1;
Grad_entropy=-sumabs(h_e.*log2(abs(h_e)));

[Gx_ms_1,Gy_ms_1] = imgradientxy(I_ms_1,'central');
f1_ms_1=sum(abs(Gy_ms_1),'all');
h_e_ms_1=abs(Gy_ms_1)./f1_ms_1;
Grad_entropy_ms_1=-sumabs(h_e_ms_1.*log2(abs(h_e_ms_1)));

[Gx_ms_2,Gy_ms_2] = imgradientxy(I_ms_2,'central');
f1_ms_2=sum(abs(Gy_ms_2),'all');
h_e_ms_2=abs(Gy_ms_2)./f1_ms_2;
Grad_entropy_ms_2=-sumabs(h_e_ms_2.*log2(abs(h_e_ms_2)));
Grad_Entropy_phantom_ms=[Grad_entropy; Grad_entropy_ms_1; Grad_entropy_ms_2];

% Laplacian 1
Laplace=[-1 -2 -1; -2 12 -2; -1 -2 -1];
k1_I=sumabs(conv2(I, Laplace, 'same'));
k1_I2=sumabs(conv2(I_ms_1, Laplace, 'same'));
k1_I3=sumabs(conv2(I_ms_2, Laplace, 'same'));
Laplacian1=[k1_I;k1_I2;k1_I3];

% Laplacian 2
Laplace2=[0 1 0; 1 -4 1; 0 1 0];
k2_I=sumabs(conv2(I, Laplace2, 'same'));
k2_I2=sumabs(conv2(I_ms_1, Laplace2, 'same'));
k2_I3=sumabs(conv2(I_ms_2, Laplace2, 'same'));
Laplacian2=[k2_I;k2_I2;k2_I3];

% Autocorrelation 1
for AC_index=1:256
    for AC_indexj=1:256
        if AC_indexj+2<256%How should I handel edge cases?
            Plus1(AC_index,AC_indexj)=I(AC_index,AC_indexj)*I(AC_index,AC_indexj+1);
            Plus1_2(AC_index,AC_indexj)=I_ms_1(AC_index,AC_indexj)*I_ms_1(AC_index,AC_indexj+1);
            Plus1_3(AC_index,AC_indexj)=I_ms_2(AC_index,AC_indexj)*I_ms_2(AC_index,AC_indexj+1);
        end
        if AC_indexj+2<256
            Plus2(AC_index,AC_indexj)=I(AC_index,AC_indexj)*I(AC_index,AC_indexj+2);
            Plus2_2(AC_index,AC_indexj)=I_ms_1(AC_index,AC_indexj)*I_ms_1(AC_index,AC_indexj+2);
            Plus2_3(AC_index,AC_indexj)=I_ms_2(AC_index,AC_indexj)*I_ms_2(AC_index,AC_indexj+2);
        end
    end    
end
Auto_correct_1=sum(I.^2,'all')-sum(Plus1,'all');
Auto_correct_1_2=sum(I_ms_1.^2,'all')-sum(Plus1_2,'all');
Auto_correct_1_3=sum(I_ms_2.^2,'all')-sum(Plus1_3,'all');
Auto_Correlation_1=[Auto_correct_1;Auto_correct_1_2;Auto_correct_1_3];

% Autocorrelation 2
Auto_correct_2=sum(Plus1,'all')-sum(Plus2,'all');
Auto_correct_2_2=sum(Plus1_2,'all')-sum(Plus2_2,'all');
Auto_correct_2_3=sum(Plus1_2,'all')-sum(Plus2_3,'all');
Auto_Correlation_2=[Auto_correct_2;Auto_correct_2_2;Auto_correct_2_3];

%Cube of Normalised intensities
CNI=sum((I./sum(I,'all'))^3,'all')
CNI_2=sum((I_ms_1./sum(I_ms_1,'all'))^3,'all')
CNI_3=sum((I_ms_2./sum(I_ms_2,'all'))^3,'all')
Cube_norm_intensities=[CNI;CNI_2;CNI_3]

%Sum of squares Entropy normalisation (David's paper)
B_max=sqrt(sum(I.^2,'all'))
Entropy_SSnorm=-sum((I./B_max)*log(I./B_max),'all')
B_max2=sqrt(sum(I_ms_1.^2,'all'))
Entropy_SSnorm2=-sum((I_ms_1./B_max2)*log(I_ms_1./B_max2),'all')
B_max3=sqrt(sum(I_ms_2.^2,'all'))
Entropy_SSnorm3=-sum((I_ms_2./B_max3)*log(I_ms_2./B_max3),'all')
Sum_of_squares_Entropy=[Entropy_SSnorm;Entropy_SSnorm2;Entropy_SSnorm3]

% Gradient
[Gx,Gy] = imgradientxy(I,'central');
f1=sum(abs(Gy),'all');% divide to normalise by mean pixel value
[Gx2,Gy2] = imgradientxy(I_ms_1,'central');
f1_2=sum(abs(Gy2),'all');% divide to normalise by mean(real(combined_image2),'all') ?
[Gx3,Gy3] = imgradientxy(I_ms_2,'central');
f1_3=sum(abs(Gy3),'all');% divide to normalise by mean pixel value
f1_grad_images=[f1; f1_2; f1_3];

% Calculate the normalised gradient squared (R13)
F13_0=sum((abs(Gy)./sum(abs(Gy),'all')).^2,'all');
F13_2=sum((abs(Gy2)./sum(abs(Gy2),'all')).^2,'all');
F13_3=sum((abs(Gy3)./sum(abs(Gy3),'all')).^2,'all');
Norm_Grad_squared=[F13_0; F13_2; F13_3];

%Creating a table summarising the metrics results
v=[{'No motion'};{'Little Motion'};{'Bigger motion'}];
Tab_metrics = table(v, f1_grad_images, Entropy2_ms, Grad_Entropy_phantom_ms, Norm_Grad_squared,Laplacian1,Laplacian2,Auto_Correlation_1,Auto_Correlation_2,Cube_norm_intensities,Sum_of_squares_Entropy)
writetable(Tab_metrics,'TablesdataMultiShot.xlsx','Sheet',1,'Range','D1')

%Calculating percentages
Entropy2=100*(Entropy2_ms-Entropy2_ms(1))./Entropy2_ms(1);
Gradient=100*(f1_grad_images-f1_grad_images(1))./f1_grad_images(1);
Grad_Entropy=100*(Grad_Entropy_phantom_ms-Grad_Entropy_phantom_ms(1))./Grad_Entropy_phantom_ms(1);
Norm_Grad_sq=100*(Norm_Grad_squared-Norm_Grad_squared(1))./Norm_Grad_squared(1);
Lapl1=100*(Laplacian1-Laplacian1(1))./Laplacian1(1);
Lapl2=100*(Laplacian2-Laplacian2(1))./Laplacian2(1);
AC1=100*(Auto_Correlation_1-Auto_Correlation_1(1))./Auto_Correlation_1(1);
AC2=100*(Auto_Correlation_2-Auto_Correlation_2(1))./Auto_Correlation_2(1);
Cube_norm=100*(Cube_norm_intensities-Cube_norm_intensities(1))./Cube_norm_intensities(1);
SoS_Entropy=100*(Sum_of_squares_Entropy-Sum_of_squares_Entropy(1))./Sum_of_squares_Entropy(1);

Tab_metrics2 = table(v, Entropy2, Gradient, Grad_Entropy, Norm_Grad_sq,Lapl1,Lapl2,AC1,AC2,Cube_norm,SoS_Entropy)
writetable(Tab_metrics2,'TablesdataMultiShot.xlsx','Sheet',2,'Range','D1')

%% See contribution to metric in blocs (compare motion and no motion)
%Gradient contribution in blocs
[Gx12,Gy12] = imgradientxy(I,'central');
h_ee2=abs(I)./sum(abs(I),'all');
max(max(h_ee2.*log2(abs(h_ee2))));
B2 = resize(mat2gray(h_ee2.*log2(abs(h_ee2))),[8,8]);
figure
subplot(1,2,1)
imshow(I,[])%make it bigger in display
title('Virtual phantom')
subplot(1,2,2)
imshow(B2,[],InitialMagnification='fit')
title('Gradient contribution in blocs')

%Entropy contribution in blocs
[h, binLocations]=imhist(I);
h(h==0) = []; %remove zero entries
h = h ./ numel(I); %normalising h

%% Creating graphs multishot realistic
for ishot = 1: nshot
    shot2k{ishot} = [ishot:lpershot:nk] ; % k-space lines in each shot
end

t2shot{1} = [1:8] ; % shots in time period 1
t2shot{2} = [9:16] ; % shots in time period 2

[X, Y] = meshgrid(1:image_size(2), 1:image_size(1));
phi_x=[];
% loop over each time period (here 2)
for index_i=0:15

    prostate_centeri = [128-index_i, 128-index_i];
    prostate_maski = ((X - prostate_centeri(1)) / prostate_axis_lengths(1)).^2 + ...
    ((Y - prostate_centeri(2)) / prostate_axis_lengths(2)).^2 <= 1;
    numerical_phantom_i = background_intensity * ones(image_size);
    %Creating motion numerical phantom
    numerical_phantom_i(body_mask) = body_intensity;
    numerical_phantom_i(prostate_maski) = prostate_intensity;
    numerical_phantom_i(rectum_mask) = rectum_intensity;
    numerical_phantom_i(bladder_mask) = bladder_intensity;
    numerical_phantom_i(bone_mask_left) = bone_intensity;
    numerical_phantom_i(bone_mask_right) = bone_intensity;
    texture_linesi=text_lines;
    texture_linesi(~prostate_maski) = 100; % Limit to prostate area
    numerical_phantom_i = numerical_phantom_i+ texture_linesi;

    
    % Computing FFT of the numerical phantom
    numerical_phantom_fft_i = fftshift(fft2(ifftshift(numerical_phantom_i)));

    for itime = 1:2
        % set the object at this time period
        if itime == 1
            shifted_objecti = numerical_phantom;
        else
            shifted_objecti = numerical_phantom_i;
        end
    
        % compute k-space of object
        kshiftedi = fftshift(fft2(ifftshift(shifted_objecti))) ;
    
        % below aims to fill combinedk with the k-space lines acquired at this
        % time period - finds the shots and the lines within the shots.
        shots = t2shot{itime} ; % the shots in this time period
    
        for ishot = 1:length(shots) % loop over the shots in this time period
            this_shot = shots(ishot) ;
            klines_this_shot = shot2k{this_shot} ; % the k-space lines in this shot
            combinedki(klines_this_shot,:) = kshiftedi(klines_this_shot,:) ;
        end

    end
    imgfinali = fftshift(ifft2(ifftshift(combinedki))) ;
    %adding noise
    realchannel = normrnd(0,N,256,256) + imgfinali;
    imaginarychannel = normrnd(0,N,256,256);
    imgfinali = sqrt(realchannel.^2 + imaginarychannel.^2);

    if index_i==0
        initial_image=imgfinali;
    end
    % Gradient Entropy (normalised)
    I_i = mat2gray(abs(imgfinali),[0,256]);
    Ge_i = entropy(I_i);
    phi_x(index_i+1) = Ge_i;

end
figure;
imshow(abs(imgfinali), []);

%Plotting the graph for variable displacement
pixel_displacement=0:15;
figure
plot(pixel_displacement,phi_x)
title('Entropy as a function of prostate displacement "Multi-shot"', 'FontSize', 20)
xlabel('Displacement of the prostate (in pixels)', 'FontSize', 18)
ylabel('Entropy quality metric', 'FontSize', 18)
set(gca, 'FontSize', 16);