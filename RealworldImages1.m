clear all
close all

%% Opening up real world images

%Open and loading images (use dicom functions)
collection_data = dicomCollection('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_shared/DataForJaime/inn129/DICOM/I12');
V12 = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_shared/DataForJaime/inn129/DICOM/I12');%motion
V14 = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_shared/DataForJaime/inn129/DICOM/I14');%no motion repeat

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
nexttile
imshow(roi_image12,[]);
title('ROI Image 12');
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
    nexttile(index)
    imshow(roi_images_12,[]);
    title(sprintf('Slice %d', index))
    hold on;
    nexttile(index+7)
    imshow(roi_images_14,[]);
    title(sprintf('Slice %d', index))

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


%Plotting other metrics
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
ax = gca;
ax.FontSize = 17; 

%% See contribution to metric in blocs
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
    
    
    %Sum of squares Entropy normalisation
    B_max_12=sqrt(sum(I_12.^2,'all'));
    B_max_hel14=sqrt(sum(I_14.^2,'all'));
    
    %Autocorrelation 1
    a=size(I_12);
    for AC_index=1:a(1)
        for AC_indexj=1:a(2)
            if AC_indexj+2<a(2)
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
    

end



%% Calculate the same metrics for the whole image
figure(102)
tl = tiledlayout(2,7,'TileSpacing','Compact');
for index=1:7
    images_12 = image_12(:, :, :,index);
    images_14= image_14(:, :, :,index);
    
    %Plot all ROI images to see where the problem can be
    nexttile(index)
    imshow(images_12,[]);
    title(sprintf('Slice %d', index))
    nexttile(index+7)
    imshow(images_14,[]);
    title(sprintf('Slice %d', index))

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

%% Opening up real world images (motion dataset)

%Inn129 (With and without motion, reaquisition)
V12 = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_shared/DataForJaime/inn129/DICOM/I12');
V14 = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_shared/DataForJaime/inn129/DICOM/I14');%no motion

%MSHP1
V21 = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_shared/DataForJaime/20240307_MSHshVPAT/Mshp1/(Not_For_Clinical_Use)_Cmi_Ingenia_Volunteer - 0/T2W_TSE_axTE110_201/IM-0001-0025.dcm');
%CLM0053
V31 = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_shared/DataForJaime/CLMRRV0053/DICOM/I9');%no motion

%Inn221
V41 = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_shared/DataForJaime/Inn221Kgo/Innovate - 0/T2W_TSE_ax_801/IM-0001-0035.dcm');

%Inn348
V51 = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_shared/DataForJaime/inn348/Inn_348/Mri_Prostate_With_Contrast - 0/T2W_TSE_ax_301');

%Hmu057
V61 = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_shared/DataForJaime/Hmu_057/HistoMri - 0/T2W_TSE_ax_401');

%Hmu254
V71 = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_shared/DataForJaime/Hmu_254/Mri_Histomap - 0/T2W_Axial_401');

%Hmu255
V81 = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_shared/DataForJaime/Hmu_255/Mri_Histomap - 0/T2W_Axial_201');

% Hmu271
V91 = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_shared/DataForJaime/Hmu_271/Mri_Histomap - 0/T2W_Axial_301/IM-0001-0026.dcm');

% Rru1033
V101 = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_shared/DataForJaime/20220824_Vp_Bio_Rru1033/Mri_Pelvis_Prostate - 0/T2W_Axial_301');

%% Manual scoring per dataset
% 2 Motion in odd slices (reacquire)
% 0 No motion
% -1 Unclear motion in some slices (reacquire everything?)
% -2 Motion in even slices (reacquire)
ManualScoring=[-2 0 -1 -1 -2 -2 -1 -1 -1 -1 -2];

%% Manual scoring for each slice
% 5 Motion (very blurry)
% 4 Motion (slighlty blurry)
% 3 Unclear/Some motion/Sharp and blurry
% 2 No Motion (not that sharp)
% 1 No Motion (very sharp)
ManualScore(1).values=[5 1 5 1 5 1 5];
ManualScore(2).values=[1 1 1 1 1 1 1];
ManualScore(3).values=[3 4 2 4 1 3 1 2 1 1 2];
ManualScore(4).values=[2 5 4 1 2 5 1 1 5];
ManualScore(5).values=[2 5 4 5 3 5 2 4 2 4];
ManualScore(6).values=[4 2 3 1 4 3 4 3 4 2 4];
ManualScore(7).values=[3 3 2 4 3 4 3 3 2 4 3];
ManualScore(8).values=[2 4 1 1 3 1 2 4 2];
ManualScore(9).values=[2 2 2 2 3 4 3 4 3 4];
ManualScore(10).values=[4 2 4 3 2 2 3 2 4 3];
ManualScore(11).values=[1 4 1 3 2 2 3 3 4 5];

%% Determining slices of interest for each dataset
M = cell(4, 1) ;
M{1} = V12(:,:,1,14:20);
M{2} = V14(:,:,1,14:20);
M{3} = V21(:,:,1,7:16);
M{4} = V31(:,:,1,10:18);

M{5} = V41(:,:,1,11:20);
M{6} = V51(:,:,1,8:18);
M{7} = V61(:,:,1,10:20);
M{8} = V71(:,:,1,10:18);
M{9} = V81(:,:,1,11:20);
M{10} = V91(:,:,1,11:20);
M{11} = V101(:,:,1,11:20);

min_max=zeros(11,2);
for gindex = 1:11
    if gindex == 1 || gindex == 2
            min_max(gindex,1)=14;
            min_max(gindex,2)=20;
        elseif gindex == 3
            min_max(gindex,1)=7;
            min_max(gindex,2)=16;
        elseif gindex == 4 || gindex == 8
            min_max(gindex,1)=10;
            min_max(gindex,2)=18;
        elseif gindex == 6
            min_max(gindex,1)=8;
            min_max(gindex,2)=18;
        elseif gindex == 7
            min_max(gindex,1)=10;
            min_max(gindex,2)=20;
        else
            min_max(gindex,1)=11;
            min_max(gindex,2)=20;  
    end
end

%% Manually defined ROI for each set of images (Done once and saved to a file and used from there)

% % Define manually a ROI for each set of images and save them
% % roi=zeros(11,4);
% for index_roi=1:11
%     image=M{index_roi}(:, :, :,7);
%     figure(15)
%     imshow(image,[])
%     r = drawrectangle();
%     roi(index_roi,:)=r.Position;
% end
% writematrix(roi,'Manual_ROIs.xls')


%% Automated Entropy and Polynomial fit calculations (+Rajski+Plotting manual scoring per slice)
roi=readmatrix('Manual_ROIs.xls');

I_roi=[];
Ge_matrix=zeros(11,11);
Lapl1=zeros(11,11);
Gradient=zeros(11,11);
residuals_matrix=zeros(11,11);
poly_matrix=zeros(11,4);
Rajski_distance=zeros(11,11);
Laplace=[-1 -2 -1; -2 12 -2; -1 -2 -1];

%looping for each dataset
for index_v=1:11
    Number_of_slices=size(M{index_v});
    x=roi(index_v,1);
    y=roi(index_v,2);
    width=roi(index_v,3);
    height=roi(index_v,4);
    for index_slice=1:Number_of_slices(4)
        roi_image = mat2gray(M{index_v}(y:y+height-1, x:x+width-1, :,:));
        I = mat2gray(M{index_v}(:, :, :,:));
        I_roi =roi_image(:, :, :,index_slice);
        
        %entropy calculation and saving it in a matrix
        Ge_matrix(index_v,index_slice) = entropy(I_roi);

        %Laplacian 1
        Lapl1(index_v,index_slice)=sumabs(conv2(I_roi, Laplace, 'same'));

        %Gradient
        I = mat2gray(I_roi);
        [Gx,Gy] = imgradientxy(I,'central');
        Gradient(index_v,index_slice)=sum(abs(Gy),'all');% divide to normalise by mean pixel value
        
        %Gradient Entropy
        h_e=abs(Gy)./Gradient(index_v,index_slice);
        Grad_entropy(index_v,index_slice)=-sumabs(h_e.*log2(abs(h_e)));

        % Autocorrelation 1
        for AC_index=1:height
            for AC_indexj=1:height
                if AC_indexj+2<width
                    Plus1(AC_index,AC_indexj)=I_roi(AC_index,AC_indexj)*I_roi(AC_index,AC_indexj+1);
                end
                if AC_indexj+2<height
                    Plus2(AC_index,AC_indexj)=I_roi(AC_index,AC_indexj)*I_roi(AC_index,AC_indexj+2);
                end
            end    
        end
        Auto_correct_1(index_v,index_slice)=sum(I.^2,'all')-sum(Plus1,'all');
        
        % Autocorrelation 2
        Auto_correct_2(index_v,index_slice)=sum(Plus1,'all')-sum(Plus2,'all');
         
        %Calculation of Rajski distance
        if index_slice ~= 1
            [MI, JE] = mutualinfo(I_roi,Previous_I_roi);
            Rajski_distance(index_v,index_slice)=1-MI/JE;
        end
        Previous_I_roi=I_roi;
        
    end

    %Creatinf the polynomial fit
    i_min=min_max(index_v,1);
    i_max=min_max(index_v,2);
    poly_matrix(index_v,:) = polyfit(i_min:i_max,Ge_matrix(index_v,1:1:index_slice),3);

    % Calculate residuals
    residuals_matrix(index_v, 1:Number_of_slices(4)) = Ge_matrix(index_v, 1:Number_of_slices(4)) - polyval(poly_matrix(index_v,:),i_min:i_max);
    
    %Plotting figure entropy
    figure(21)
    subplot(3,4,index_v)
    plot(i_min:i_max,Ge_matrix(index_v,1:length(i_min:i_max)))
    hold on;
    title('Plot of Entropy for each slice', index_v)
    xx=linspace(i_min,i_max);
    plot(xx,polyval(poly_matrix(index_v,:),xx),'LineWidth',2)
    ax = gca;
    ax.FontSize = 17;
    xlabel('Slice number')
    ylabel('Entropy')
    hold off;

    %Calculating Average distance from polynomial
    average_distance=10*sum(abs(polyval(poly_matrix(index_v,:),i_min:i_max)-Ge_matrix(index_v,1:length(i_min:i_max))))/length(i_min:i_max);
    Matrix_av_d(index_v)=average_distance;
    
  
    %Plotting manual scoring for each slice
    figure(23)
    subplot(3,4,index_v)
    N=length(i_min:i_max);
    c=zeros(N,3); 
    Manual_Scoring= ManualScore(index_v).values;
    for i=1:N
        if Manual_Scoring(i)>4 
            c(i,:)=[1,0,0]; 
        elseif Manual_Scoring(i)==4 
            c(i,:)=[1,0.7,0]; 
        elseif Manual_Scoring(i)==3 
            c(i,:)=[1,1,0]; 
        elseif Manual_Scoring(i)==2
            c(i,:)=[0.7,1,0]; 
        elseif Manual_Scoring(i)==1
            c(i,:)=[0,1,0]; 
        else
            c(i,:)=[0,0,0]; 
        end
    end
    scatter(i_min:i_max,Ge_matrix(index_v,1:length(i_min:i_max)),150,c,'filled'); hold on
    plot(i_min:i_max,Ge_matrix(index_v,1:length(i_min:i_max)),'-k'); hold on  %add line connecting the points, as you requested
    xx=linspace(i_min,i_max);
    plot(xx,polyval(poly_matrix(index_v,:),xx),'LineWidth',2);
    ax = gca;
    ax.FontSize = 17;
    hold off
    xlabel('Slice number')
    ylabel('Entropy')



    %Plotting other metrics
    %Plotting Lapl1
    figure(24)
    subplot(3,4,index_v)
    N=length(i_min:i_max);
    c=zeros(N,3); 
    Manual_Scoring= ManualScore(index_v).values;
    for i=1:N
        if Manual_Scoring(i)>4 
            c(i,:)=[1,0,0]; 
        elseif Manual_Scoring(i)==4 
            c(i,:)=[1,0.7,0]; 
        elseif Manual_Scoring(i)==3 
            c(i,:)=[1,1,0]; 
        elseif Manual_Scoring(i)==2
            c(i,:)=[0.7,1,0]; 
        elseif Manual_Scoring(i)==1
            c(i,:)=[0,1,0]; 
        else
            c(i,:)=[0,0,0]; 
        end
    end
    scatter(i_min:i_max,Lapl1(index_v,1:length(i_min:i_max)),100,c,'filled'); hold on
    plot(i_min:i_max,Lapl1(index_v,1:length(i_min:i_max)),'-k'); hold on 
    hold off
    xlabel('Slice number')
    ylabel('Laplacian')

    
    %Plotting Gradient
    figure(25)
    subplot(3,4,index_v)
    scatter(i_min:i_max,Gradient(index_v,1:length(i_min:i_max)),100,c,'filled'); hold on
    plot(i_min:i_max,Gradient(index_v,1:length(i_min:i_max)),'-k'); hold on  
    hold off
    xlabel('Slice number')
    ylabel('Gradient')

    %Plotting Gradient Entropy
    figure(26)
    subplot(3,4,index_v)
    scatter(i_min:i_max,Grad_entropy(index_v,1:length(i_min:i_max)),100,c,'filled'); hold on
    plot(i_min:i_max,Grad_entropy(index_v,1:length(i_min:i_max)),'-k'); hold on  
    hold off
    xlabel('Slice number')
    ylabel('Gradient Entropy')

    %Plotting AC1
    figure(27)
    subplot(3,4,index_v)
    scatter(i_min:i_max,Auto_correct_1(index_v,1:length(i_min:i_max)),100,c,'filled'); hold on
    plot(i_min:i_max,Auto_correct_1(index_v,1:length(i_min:i_max)),'-k'); hold on  
    hold off
    xlabel('Slice number')
    ylabel('AC1')

    %Plotting AC2
    figure(28)
    subplot(3,4,index_v)
    scatter(i_min:i_max,Auto_correct_2(index_v,1:length(i_min:i_max)),100,c,'filled'); hold on
    plot(i_min:i_max,Auto_correct_2(index_v,1:length(i_min:i_max)),'-k'); hold on  
    hold off
    xlabel('Slice number')
    ylabel('AC2')

end


%% Proposed metric based on Polynomial fit

%Creating a method to determine if values are above or below polynomial fit

% Determining if all odd and even real values are above or below the polynomial fit
for index_v=1:11    
    i_min=min_max(index_v,1);
    i_max=min_max(index_v,2);
    slices=i_min:i_max;
    odd_above = all(Ge_matrix(index_v, 1:2:length(slices)) > polyval(poly_matrix(index_v,:),slices(1:2:end)));
    even_above = all(Ge_matrix(index_v, 2:2:length(slices)) > polyval(poly_matrix(index_v,:),slices(2:2:end)));
    odd_below = all(Ge_matrix(index_v, 1:2:length(slices)) < polyval(poly_matrix(index_v,:),slices(1:2:end)));
    even_below = all(Ge_matrix(index_v, 2:2:length(slices)) < polyval(poly_matrix(index_v,:),slices(2:2:end)));
    
    % Save results in status vectors
    if odd_above
        odd_status_vector(index_v) = 1;
    elseif odd_below
        odd_status_vector(index_v) = -1;
    else
        odd_status_vector(index_v) = 0; % Mixed values
    end
    
    if even_above
        even_status_vector(index_v) = 1;
    elseif even_below
        even_status_vector(index_v) = -1;
    else
        even_status_vector(index_v) = 0; % Mixed values
    end
end


%% Automated scoring
for index_v = 1:11
    if rem(min_max(index_v,1),2) == 0 
        for pol_index = min_max(index_v,1):2:min_max(index_v,1)
            
        end
        for pol_index = min_max(index_v,1)+1:2:min_max(index_v,1)
            
        end 
    else
        
    end
end

%% Opening up real world images (Validation dataset)

%CLMRRV0039 (Motion)
V1_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0039/Clmrrv - 727347299/T2W_Axial_401');

%CLMRRV0037 (Motion)
V2_NM =dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0037/Clmrrv - 726997249/T2W_Axial_401');

%CLMRRV0070 (No Motion)
V3_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0070/Clmrrv - 726312723/T2W_Axial_401');

%CLMRRV0008 (Motion)
V4_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0008/Rrv - 726491491/T2W_Axial_401');

%CLMRRV0032 (Motion)
V5_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0032/Clmrrv - 727774865/T2W_Axial_401');

%CLMRRV0029 (Motion)
V6_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0029/Clmrrv - 727172670/T2W_Axial_401');

%CLMRRV0012 (Unclear)
V7_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0012/Clmrrv - 727432854/T2W_Axial_401');

%CLMRRV0021 (Motion)
V8_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0021/Clmrrv - 725618838/T2W_Axial_401');

%CLMRRV0035 (Motion)
V9_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0035/Clmrrv - 725619497/T2W_Axial_501');

%CLMRRV0026 (Unclear)
V10_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0026/Clmrrv - 726572248/T2W_Axial_401');

%CLMRRV0034 (Motion)
V11_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0034/Clmrrv - 725614925/T2W_Axial_401');

%HMU 253 (Motion)
V12_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_withheld/Hmu_253/Mri_Histomap - 0/T2W_Axial_501');

%HMU 283 (Unclear)
V13_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_withheld/Hmu_283/Mri_Histomap - 0/T2W_Axial_301');

%HMU 279 (Unclear)
V14_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_withheld/Hmu_279/Mri_Histomap - 0/T2W_Axial_301');

%HMU 280 (Motion)
V15_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_withheld/Hmu_280/Mri_Histomap - 0/T2W_Axial_301');

%HMU 281 (Unclear)
V16_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_withheld/Hmu_281/Mri_Histomap - 0/T2W_Axial_301');

%PSEQ 011 (Motion)
V17_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_withheld/Pseq011/(Not_For_Clinical_Use)_Cmi_Ingenia_Volunteer - 0/T2W_Axial_asym_27slFIDthro_901');

%CLMRRV0171 (No Motion)
V18_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0171/Clmrrv - 190084743/T2W_axial_401');

%CLMRRV0167 (No Motion)
V19_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0167/Clmrrv - 757412155/T2W_axial_201');

%CLMRRV0077(No Motion)
V20_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0077/Clmrrv - 726142595/T2W_Axial_401');

%CLMRRV0162 (No Motion)
V21_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0162/Clmrrv - 757158808/T2W_Axial_401');

%CLMRRV0073 (No Motion)
V22_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0073/Clmrrv - 727609868/T2W_Axial_401');

%CLMRRV0050 (No Motion)
V23_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0050/Clmrrv - 727352003/T2W_Axial_301');

%CLMRRV0033 (No Motion)
V24_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0033/Clmrrv - 727778903/T2W_Axial_401');

%CLMRRV0072 (Unclear)
V25_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0072/Clmrrv - 727521541/T2W_Axial_401');

%CLMRRV0025 (No Motion)
V26_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0025/Clmrrv - 726567864/T2W_Axial_401');

%CLMRRV0027 (No Motion)
V27_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0027/Clmrrv - 726831881/T2W_Axial_501');

%CLMRRV0019 (No Motion)
V28_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0019/Clmrrv - 727265812/T2W_Axial_401');

%CLMRRV0001 (No Motion)
V29_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0001/Mri_Prostate_With_Contrast - 0/T2W_Axial_401');

%CLMRRV0018 (No Motion)
V30_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0018/Clmrrv - 726844089/T2W_Axial_401');

%CLMRRV0015 (No Motion)
V31_NM = dicomreadVolume('/Users/jaime/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/Dissertation/Withheld data/Clmrrv0015/Clmrrv - 725963907/T2W_Axial_401');

%% Manual scoring per dataset
% 2 Motion in some slices (reacquire)
% 0 No motion
% -1 Unclear motion in some slices (reacquire everything?)

%% Determine slices of interest for each dataset

M = cell(4, 1) ;
M{1} = V1_NM(:,:,1,7:17);
M{2} = V2_NM(:,:,1,10:18);
M{3} = V3_NM(:,:,1,9:17);
M{4} = V4_NM(:,:,1,11:19);
M{5} = V5_NM(:,:,1,11:19);
M{6} = V6_NM(:,:,1,9:20);
M{7} = V7_NM(:,:,1,8:18);
M{8} = V8_NM(:,:,1,7:18);
M{9} = V9_NM(:,:,1,6:21);
M{10} = V10_NM(:,:,1,13:20);
M{11} = V11_NM(:,:,1,11:19);
M{12} = V12_NM(:,:,1,10:16);
M{13} = V13_NM(:,:,1,9:18);
M{14} = V14_NM(:,:,1,10:19);
M{15} = V15_NM(:,:,1,12:22);
M{16} = V16_NM(:,:,1,9:17);
M{17} = V17_NM(:,:,1,9:20);
M{18} = V18_NM(:,:,1,12:25);
M{19} = V19_NM(:,:,1,26:36);
M{20} = V20_NM(:,:,1,11:19);
M{21} = V21_NM(:,:,1,6:20);
M{22} = V22_NM(:,:,1,8:16);
M{23} = V23_NM(:,:,1,11:22);
M{24} = V24_NM(:,:,1,11:17);
M{25} = V25_NM(:,:,1,10:18);
M{26} = V26_NM(:,:,1,6:20);
M{27} = V27_NM(:,:,1,8:15);
M{28} = V28_NM(:,:,1,12:19);
M{29} = V29_NM(:,:,1,7:18);
M{30} = V30_NM(:,:,1,12:17);
M{31} = V31_NM(:,:,1,14:21);


min_max=zeros(31,2);
for gindex = 1:31
    if gindex == 1 
            min_max(gindex,1)=7;
            min_max(gindex,2)=17;
    elseif gindex == 2 || gindex == 25
            min_max(gindex,1)=10;
            min_max(gindex,2)=18;    
    elseif gindex == 3 || gindex == 16
            min_max(gindex,1)=9;
            min_max(gindex,2)=17;
    elseif gindex == 4 || gindex == 5 || gindex == 11 || gindex == 20
        min_max(gindex,1)=11;
        min_max(gindex,2)=19;
    elseif gindex == 6 || gindex == 17
        min_max(gindex,1)=9;
        min_max(gindex,2)=20;
    elseif gindex == 8 || gindex == 29
        min_max(gindex,1)=7;
        min_max(gindex,2)=18;
    elseif gindex == 7
        min_max(gindex,1)=8;
        min_max(gindex,2)=18;
    elseif gindex == 9
        min_max(gindex,1)=6;
        min_max(gindex,2)=21;
    elseif gindex == 10
        min_max(gindex,1)=13;
        min_max(gindex,2)=20;
    elseif gindex == 12
        min_max(gindex,1)=10;
        min_max(gindex,2)=16;
    elseif gindex == 13
        min_max(gindex,1)=9;
        min_max(gindex,2)=18;
    elseif gindex == 14
        min_max(gindex,1)=10;
        min_max(gindex,2)=19;
    elseif gindex == 15
        min_max(gindex,1)=12;
        min_max(gindex,2)=22;
    elseif gindex == 18
        min_max(gindex,1)=12;
        min_max(gindex,2)=25;
    elseif gindex == 19
        min_max(gindex,1)=26;
        min_max(gindex,2)=36;
    elseif gindex == 21 || gindex == 26
        min_max(gindex,1)=6;
        min_max(gindex,2)=20;
    elseif gindex == 22
        min_max(gindex,1)=8;
        min_max(gindex,2)=16;
    elseif gindex == 23
        min_max(gindex,1)=11;
        min_max(gindex,2)=22;
    elseif gindex == 24
        min_max(gindex,1)=11;
        min_max(gindex,2)=17;
    elseif gindex == 27
        min_max(gindex,1)=8;
        min_max(gindex,2)=15;
    elseif gindex == 28
        min_max(gindex,1)=12;
        min_max(gindex,2)=19;
    elseif gindex == 30
        min_max(gindex,1)=12;
        min_max(gindex,2)=17;
    else
        min_max(gindex,1)=14;
        min_max(gindex,2)=21;  
    end
end

%% Previously defined ROI for each set of images (Done once and saved on a file and used it from there)

% % % Define manually a ROI for each set of images and save them
% for index_roi=18
%     image=M{index_roi}(:, :, :,1);
%     figure(15)
%     imshow(image,[])
%     r = drawrectangle();
%     roi(index_roi,:)=r.Position;
% end
% writematrix(roi,'Manual_ROIs_Withheld.xls')

%% Automated Entropy and Polynomial fit calculations (+Rajski+Plotting manual scoring per slice)
roi=readmatrix('Manual_ROIs_Withheld.xls');

I_roi=[];
Ge_matrix=zeros(10,31);
residuals_matrix=zeros(11,11);
poly_matrix=zeros(31,4);
Matrix_av_d=zeros(1,31);

%looping for each dataset
for index_v=1:31
    Number_of_slices=size(M{index_v});
    x=roi(index_v,1);
    y=roi(index_v,2);
    width=roi(index_v,3);
    height=roi(index_v,4);
    for index_slice=1:Number_of_slices(4)
        roi_image = mat2gray(M{index_v}(y:y+height-1, x:x+width-1, :,:));
        I_roi =roi_image(:, :, :,index_slice);

        %entropy calculation and saving it in a matrix
        Ge_matrix(index_v,index_slice) = entropy(I_roi);

        %Calculation of Rajski distance
        if index_slice ~= 1
            [MI, JE] = mutualinfo(I_roi,Previous_I_roi);
            Rajski_distance(index_v,index_slice)=1-MI/JE;
        end
        Previous_I_roi=I_roi;
        
    end

    %Creating the polynomial fit
    i_min=min_max(index_v,1);
    i_max=min_max(index_v,2);
    poly_matrix(index_v,:) = polyfit(i_min:i_max,Ge_matrix(index_v,1:index_slice),3);

    % Calculate residuals
    residuals_matrix(index_v, 1:Number_of_slices(4)) = Ge_matrix(index_v, 1:Number_of_slices(4)) - polyval(poly_matrix(index_v,:),i_min:i_max);
    
    %Plotting figure entropy
    figure(221)
    subplot(5,7,index_v)
    plot(i_min:i_max,Ge_matrix(index_v,1:length(i_min:i_max)))
    hold on;
    title('Entropy', index_v)
    xx=linspace(i_min,i_max);
    plot(xx,polyval(poly_matrix(index_v,:),xx))
    ax = gca;
    ax.FontSize = 17;

    %Calculating Average distance from polynomial
    average_distance=10*sum(abs(polyval(poly_matrix(index_v,:),i_min:i_max)-Ge_matrix(index_v,1:length(i_min:i_max))))/length(i_min:i_max);
    Matrix_av_d(index_v)=average_distance;
    

    % %Plotting manual scoring for each slice
    % figure(23)
    % subplot(3,4,index_v)
    % N=length(i_min:i_max);
    % c=zeros(N,3); %allocate colors
    % Manual_Scoring= ManualScore(index_v).values;
    % for i=1:N
    %     if Manual_Scoring(i)>4 
    %         c(i,:)=[1,0,0]; 
    %     elseif Manual_Scoring(i)==4 
    %         c(i,:)=[1,0.7,0]; 
    %     elseif Manual_Scoring(i)==3 
    %         c(i,:)=[1,1,0]; 
    %     elseif Manual_Scoring(i)==2
    %         c(i,:)=[0.7,1,0]; 
    %     elseif Manual_Scoring(i)==1
    %         c(i,:)=[0,1,0]; 
    %     else
    %         c(i,:)=[0,0,0]; 
    %     end
    % end
    % scatter(i_min:i_max,Ge_matrix(index_v,1:length(i_min:i_max)),150,c,'filled'); hold on
    % plot(i_min:i_max,Ge_matrix(index_v,1:length(i_min:i_max)),'-k'); hold on  %add line connecting the points, as you requested
    % xx=linspace(i_min,i_max);
    % plot(xx,polyval(poly_matrix(index_v,:),xx)); hold off
    % xlabel('Slice number')
    % ylabel('Entropy')
    % %legend('Manual scoring','Entropy plot for each slice','Polynomial fit')

end