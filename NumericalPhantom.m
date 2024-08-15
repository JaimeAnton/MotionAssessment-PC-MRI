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


% %Taking even and odd elements of images
% combined_image2 = zeros(size(numerical_phantom));% Initializing the new image
% 
% for row = 1:size(numerical_phantom, 1)
%     if mod(row, 2) == 1 %Calculating the modulus to keep only odd rows from numerical_phantom
%         combined_image2(row, :) = numerical_phantom(row, :);
%     else %Keeping the even rows from numerical_phantom2
%         combined_image2(row, :) = numerical_phantom2(row, :);
%     end
% end

% Display the combined images
figure;
subplot(1, 2, 1);
imshow(abs(combined_image2), []);
title('Combined Numerical Phantom 2 (halfs)');
subplot(1, 2, 2);
imshow(abs(combined_image3), []);
title('Combined Numerical Phantom 3 (halfs)');
% imshow(combined_image2, []);
% title('Combined Numerical Phantoms (Odd and even rows)');

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


%Creating a graph showing the displacement as a function of time (abrupt change)


%Graph showing proportion of rows from each image (linear relationship)


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

%%
% figure
% imshow(I3,[])
%%

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

%Possible normalisation strategies of the gradient





% grad=[1;0;-1];
% gradient_matrix=zeros(256,256);
% for i=1:256
%     for j=1:256
%         gradient_matrix(i,j)=sum(abs(grad*numerical_phantom_fft(i,j)));
%     end
% end
% numerical_phantom_corrected = ifftshift(ifft2(fftshift(gradient_matrix)));


% figure;
% subplot(1, 4, 1);
% imshow(numerical_phantom, []);
% title('Original Numerical Phantom');
% subplot(1, 4, 2);
% imshow(numerical_phantom_corrected, []);
% title('Motion corrected phantom');
% subplot(1, 4, 3);help 
% imagesc(log(abs(numerical_phantom_fft) + 1));
% colormap('gray');
% axis square;
% title('Original FFT phantom');
% subplot(1, 4, 4);
% imagesc(log(abs(gradient_matrix) + 1));
% colormap('gray');
% axis square;
% title('Motion corrected fft phantom');


% Display the gradient magnitude image
figure;
subplot(1,3,1)
imshow(Gy);
title('Gradient of Numerical Phantom no motion');
subplot(1,3,2)
imshow(Gy2);
title('Gradient of Numerical Phantom 2');
subplot(1,3,3)
imshow(Gy3);
title('Gradient of Numerical Phantom 3');

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
%plotting it to see what contributes to the metric
figure
subplot(1,3,1)
imshow(h_e.*log2(abs(h_e)),[])
title('No motion')
subplot(1,3,2)
imshow(h_e2.*log2(abs(h_e2)),[])
title('Some motion')
subplot(1,3,3)
imshow(h_e3.*log2(abs(h_e3)),[])
title('More motion')

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
%Present and compare data(export to excel?)

%Calculating how each metric is performing to compare them (calculate M?)


%% Rajski cost function
Rajski_CF=1-I./Ge;
Rajski_CF2=1-I2./Ge2;
Rajski_CF3=1-I3./Ge3;

%% Opening up real world images

%Open and loading images (use dicom functions)
collection_data = dicomCollection('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_shared/DataForJaime/inn129/DICOM/I12');
%Using dicombrowser to export the volume to workspace (V)
%dicomBrowser('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_shared/DataForJaime/inn129/DICOM/I12')
%Displaying image on MATLAB, The dimensions of V are [rows, columns, samples, slices]
V12 = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_shared/DataForJaime/inn129/DICOM/I12');
V14 = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_shared/DataForJaime/inn129/DICOM/I14');%no motion

figure;
image1=V12(:,:,1,20);
image2=V14(:,:,1,20);
subplot(1,2,1)
imshow(image1,[])
subplot(1,2,2)
imshow(image2,[])


%Define a ROI for each image or set of images
r1 = drawrectangle();
position=r1.Position;
%Convert the position to integer coordinates
x = round(position(1));
y = round(position(2));
width = round(position(3));
height = round(position(4));
% Create a binary mask of the ROI
mask = createMask(r1);
% Extract the ROI from the original image
roi_image12 = image1(y:y+height-1, x:x+width-1, :);
roi_image14 = image2(y:y+height-1, x:x+width-1, :);
% Display the ROI image
figure;
tiledlayout(1,2,'TileSpacing','Compact');
%subplot(1,2,1)
nexttile
imshow(roi_image12,[]);
title('ROI Image 12');
%subplot(1,2,2)
nexttile
imshow(roi_image14,[]);
title('ROI Image 14');



%% Evaluate motion metrics in real world data (repeat previous metrics on MRI images)
% Gradient Entropy (normalised)
I_inn129_12 = mat2gray((roi_image12),[0,256]);
Ge_inn129_12 = entropy(I_inn129_12);
I_inn129_14 = mat2gray((roi_image14),[0,256]);
Ge_inn129_14 = entropy(I_inn129_14);

%Iterating over volume (from image 14 to 20)
image_12=V12(:,:,1,14:20);
image_14=V14(:,:,1,14:20);

Ge_12=[];
Ge_14=[];
Lapl1_12=[];
Lapl2_12=[];


Laplace=[-1 -2 -1; -2 12 -2; -1 -2 -1];
Laplace2=[0 1 0; 1 -4 1; 0 1 0];
I_14=[];
figure(100)
tiledlayout(2,7,'TileSpacing','Compact');
for index=1:7
    roi_images_12 = image_12(y:y+height-1, x:x+width-1, :,index);
    roi_images_14= image_14(y:y+height-1, x:x+width-1, :,index);
    
    %Plot all ROI images to see where the problem can be
    %subplot(2,7,index)
    nexttile(index)
    imshow(roi_images_12,[]);
    title(sprintf('Slice %d', index))
    hold on;
    %subplot(2,7,index+7)
    nexttile(index+7)
    imshow(roi_images_14,[]);
    title(sprintf('Slice %d', index))
    %h1 = text(-0.25, 0.5,'row 1');

    I_12 = mat2gray((roi_images_12),[0,512]);
    Ge_12(index) = entropy(I_12);
    Lapl1_12(index)=sumabs(conv2(I_12, Laplace, 'same'));
    Lapl2_12(index)=sumabs(conv2(I_12, Laplace2, 'same'));
    
    [Gx,Gy] = imgradientxy(I_12,'central');
    f1=sum(abs(Gy),'all');
    h_e=abs(Gy)./f1;
    Grad_entropy_12(index)=-sumabs(h_e.*log2(abs(h_e)));
    %Calculating the normalised gradient square
    Norm_Grad_sq_12(index)=sum((abs(Gy)./sum(abs(Gy),'all')).^2,'all');

    I_14 = mat2gray((roi_images_14),[0,512]);
    Ge_14(index) = entropy(I_14);
    Lapl1_14(index)=sumabs(conv2(I_14, Laplace, 'same'));
    Lapl2_14(index)=sumabs(conv2(I_14, Laplace2, 'same'));

    [Gx,Gy] = imgradientxy(I_14,'central');
    f1=sum(abs(Gy),'all');
    h_e=abs(Gy)./f1;
    Grad_entropy_14(index)=-sumabs(h_e.*log2(abs(h_e)));
    Norm_Grad_sq_14(index)=sum((abs(Gy)./sum(abs(Gy),'all')).^2,'all');





end

%Plotting those examples
figure(101)
plot(14:20,Ge_12)
hold;
plot(14:20,Ge_14)
title('Plot of Entropy for each slice')
xlabel('Slice number')
ylabel('Entropy')
ax = gca;
ax.FontSize = 17; 

%put a circle to indicate each motion in the plot
X=[14 16 18 20];
Y=[Ge_12(1) Ge_12(3) Ge_12(5) Ge_12(7)];
plot(X,Y,'o','MarkerEdgeColor','k','MarkerSize',15)


%Plotting other metrics (Laplacian, Grad entropy, AC2)
figure(102)
plot(14:20,Lapl1_12)
hold on;
plot(14:20,Lapl1_14)
hold on;
title('Plot of Laplace 1 metric for each slice')
xlabel('Slice number')
ylabel('Laplcae 1 metric')
Y=[Lapl1_12(1) Lapl1_12(3) Lapl1_12(5) Lapl1_12(7)];
plot(X,Y,'o','MarkerEdgeColor','k','MarkerSize',15)
ax = gca;
ax.FontSize = 17; 

% figure(103)
% plot(14:20,Lapl2_12)
% hold;
% plot(14:20,Lapl2_14)
% title('Plot of Laplace 2 metric for each slice')
% xlabel('Slice number')
% ylabel('Laplcae 2 metric')

figure(104)
plot(14:20,Grad_entropy_12)
hold;
plot(14:20,Grad_entropy_14)
title('Plot of Gradient Entropy metric for each slice')
xlabel('Slice number')
ylabel('Gradient entropy metric')
X=[14 16 18 20];
Y=[Grad_entropy_12(1) Grad_entropy_12(3) Grad_entropy_12(5) Grad_entropy_12(7)];
plot(X,Y,'o','MarkerEdgeColor','k','MarkerSize',15)
% legend('I12','I14','motion')
ax = gca;
ax.FontSize = 17; 

figure(105)
plot(14:20,Norm_Grad_sq_12)
hold;
plot(14:20,Norm_Grad_sq_14)
title('Plot of Normalised Gradient squared for each slice')
xlabel('Slice number')
ylabel('Normalised Gradient squared metric')
X=[14 16 18 20];
Y=[Norm_Grad_sq_12(1) Norm_Grad_sq_12(3) Norm_Grad_sq_12(5) Norm_Grad_sq_12(7)];
plot(X,Y,'o','MarkerEdgeColor','k','MarkerSize',15)
% legend('I12','I14','motion')
ax = gca;
ax.FontSize = 17; 
%% See contribution to metric in blocs (compare motion and no motion)
%Gradient contribution in blocs
[Gx12,Gy12] = imgradientxy(I_12,'central');
h_ee2=abs(I_12)./sum(abs(I_12),'all');
max(max(h_ee2.*log2(abs(h_ee2))));
B2 = mat2gray(h_ee2.*log2(abs(h_ee2)));
figure
imshow(B2,[],InitialMagnification='fit')%make it bigger in display

figure
imshow(I_12,[])
%% use other metrics for same ROI
Laplace=[-1 -2 -1; -2 12 -2; -1 -2 -1];
Laplace2=[0 1 0; 1 -4 1; 0 1 0];


Metrics_12=zeros(7,8);
Metrics_14=zeros(7,8);
for index=1:7
    roi_images_12 = image_12(y:y+height-1, x:x+width-1, :,index);
    roi_images_14= image_14(y:y+height-1, x:x+width-1, :,index);
    I_12 = mat2gray((roi_images_12),[0,512]);
    I_14 = mat2gray((roi_images_14),[0,512]);
    [Gx_12,Gy_12] = imgradientxy(I_12,'central');
    [Gx_14,Gy_14] = imgradientxy(I_14,'central');
    %Normalised Gradient Squared
    Metrics_12(index,1) = sum((abs(Gy_12)./sum(abs(Gy_12),'all')).^2,'all');
    Metrics_14(index,1) = sum((abs(Gy_14)./sum(abs(Gy_14),'all')).^2,'all');
    % Laplacian 1
    Metrics_12(index,2) = sumabs(conv2(I_12, Laplace, 'same'));
    Metrics_14(index,2) = sumabs(conv2(I_14, Laplace, 'same'));
    % Laplacian 2
    Metrics_12(index,3) = sumabs(conv2(I_12, Laplace2, 'same'));
    Metrics_14(index,3) = sumabs(conv2(I_14, Laplace2, 'same'));
    
    %Cube of Normalised intensities,only for squared
    %matrix?
    %Metrics_12(index,4) = sum((I_12./sum(I_12,'all'))^3,'all');
    %Metrics_14(index,4) = sum((I_14./sum(I_14,'all'))^3,'all');
    
    %Sum of squares Entropy normalisation (David's paper), only for squared
    %matrix?
    B_max_12=sqrt(sum(I_12.^2,'all'));
    %Metrics_12(index,5) =-sum((I_12./B_max_12)*log(I_12./B_max_12),'all');
    B_max_hel14=sqrt(sum(I_14.^2,'all'));
    %Metrics_14(index,5) =-sum((I_14./B_max_14)*log(I_14./B_max_14),'all');
    
    %Autocorrelation 1
    a=size(I_12);
    for AC_index=1:a(1)
        for AC_indexj=1:a(2)
            if AC_indexj+2<a(2)%How should I handle edge cases?
                Plus1_12(AC_index,AC_indexj)=I_12(AC_index,AC_indexj)*I_12(AC_index,AC_indexj+1);
                Plus1_14(AC_index,AC_indexj)=I_14(AC_index,AC_indexj)*I_14(AC_index,AC_indexj+1);
            end
            if AC_indexj+2<a(2)
                Plus2_12(AC_index,AC_indexj)=I_12(AC_index,AC_indexj)*I_12(AC_index,AC_indexj+2);
                Plus2_14(AC_index,AC_indexj)=I_14(AC_index,AC_indexj)*I_14(AC_index,AC_indexj+2);
            end
        end    
    end
    Metrics_12(index,6)=sum(I_12.^2,'all')-sum(Plus1_12,'all');
    Metrics_14(index,6)=sum(I_14.^2,'all')-sum(Plus1_14,'all');
    %Autocorrelation 2
    Metrics_12(index,7) = sum(Plus1_12,'all')-sum(Plus2_12,'all');
    Metrics_14(index,7) = sum(Plus1_14,'all')-sum(Plus2_14,'all');
    
    %Gradient

end
%plot all metrics in a unique graph

%Plot slice vs motion score? How to calculate the motion score?


%% Calculate the same metrics for the whole image
figure(102)
tl = tiledlayout(2,7,'TileSpacing','Compact');
for index=1:7
    images_12 = image_12(:, :, :,index);
    images_14= image_14(:, :, :,index);
    
    %Plot all ROI images to see where the problem can be
    %subplot(2,7,index)
    nexttile(index)
    imshow(images_12,[]);
    title(sprintf('Slice %d', index))
    %subplot(2,7,index+7)
    nexttile(index+7)
    imshow(images_14,[]);
    title(sprintf('Slice %d', index))
    %h1 = text(-0.25, 0.5,'row 1');

    I_im_12 = mat2gray(images_12,[0,512]);
    Ge_im_12(index) = entropy(I_im_12);
    I_im_14 = mat2gray(images_14,[0,512]);
    Ge_im_14(index) = entropy(I_im_14);
end
%Plotting those examples
figure(103)
plot(14:20,Ge_im_12)
hold;
plot(14:20,Ge_im_14)
title('Plot of Entropy for each slice')
xlabel('Slice number')
ylabel('Entropy')

%put a circle to indicate each motion in the plot
X1=[14 16 18 20];
Y1=[Ge_im_12(1) Ge_im_12(3) Ge_im_12(5) Ge_im_12(7)];
plot(X1,Y1,'o','MarkerEdgeColor','k','MarkerSize',15)
legend('I12','I14','motion')

%Creating a table summarising the metrics results
v1=transpose(14:20);