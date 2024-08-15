%% Opening up real world images

%Open and loading images (use dicom functions)
collection_data = dicomCollection('/Users/jaime/Library/CloudStorage/OneDrive-SharedLibraries-UniversityCollegeLondon/Atkinson, David - Jaime_shared/DataForJaime/inn129/DICOM/I12');

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