close all;clear all;
AI=imread('./Original_Image_path/xxx.jpg');
I=rgb2gray(AI);
s= rng('default');%rng(seed) specifies the seed for the random number generator
k=256;sigma = 20;
I=imresize(I,[k k]);
B = double(I) + sigma * randn(size(I));
 imshow(uint8(B),[]);title('Noisy image');
%-------------------------------------------patameters
n=8;%size of each atom
N=k;
np=(N-n+1)^2;%number of patched
pat= zeros(n^2,np);
ix=1;
id = 1:k^2;Indx=reshape(id,k,k);
%--------------patch generation
for i=1:(k-n+1)
    for j=1:(k-n+1)
        temp1=B(i:i+n-1, j:j+n-1);
        temp2=Indx(i:i+n-1, j:j+n-1);
        pat(:,ix)=reshape(temp1,[],1);
        patIndx(:,ix)=reshape(temp2,[],1);
        ix=ix+1;

    end  
end

%--------------------------------------Encryption


[Mx,Nx] = size(pat);
cols = size(pat,2);
U = randperm(cols);%U=1:cols;
permuted_pat = pat(:,U);
permuted_patIndx =  patIndx(:,U);

Q= orth(randn(n^2));  %%% second key (q)
Pat_Enc = Q*permuted_pat;

%-------------------------------MOD DL-----------------------------
%---------------------------------------------------------------------
%---------------------------------------------------------------------
%-------------dictionary learning
param.K = 3*n^2;% number of atom in dictionary
param.L = 6;%order of sparsity
param.numIteration = 50;
param.errorFlag = 0;param.preserveDCAtom =0;
param.TrueDictionary = 0;
param.displayProgress = 1;
% param.InitializationMethod = 'DataElements';
param.InitializationMethod = 'GivenMatrix';
param.initialDictionary = pat(:,1:param.K);%initial dictionary

display('Training dictionary  on plain data...')
tic
[DicMod, outputMod] = MOD(pat,param);
toc

% param.InitializationMethod = 'DataElements';
param.InitializationMethod = 'GivenMatrix';
param.initialDictionary = Q * pat(:,1:param.K);%initial dictionary

tic
display('Training dictionary on encrypted data ...')
[D_EN, outputMod_EN] = MOD(Pat_Enc,param);
toc
RecPat= DicMod*outputMod.CoefMatrix;
RecPatDec= Q'*D_EN*outputMod_EN.CoefMatrix;
RecPatEn= D_EN*outputMod_EN.CoefMatrix;

%---------------------------------------------------------------------
%---------------------------------------------------------------------
%---------------------------------------------------------------------
%-------------------reverse patch generation
temppat=reshape(RecPat,[],1);
temppat_Dec=reshape(RecPatDec,[],1);
temppat_Enc=reshape(RecPatEn,[],1);

temppatidx_dec=reshape(permuted_patIndx,[],1);
temppatidx=reshape(patIndx,[],1);

ReImage = zeros(k);
ReImage_Dec = zeros(k);
ReImage_Enc = zeros(k);


for i=1:k
    i
    for j = 1:k
        pix = k*(j-1)+i;
        xx_dec = find(temppatidx_dec==pix);
        ReImage_Dec(i,j) = mean(temppat_Dec(xx_dec));

        xx = find(temppatidx==pix);
        ReImage(i,j) = mean(temppat(xx));
        ReImage_Enc(i,j) = mean(temppat_Enc(xx));

    end  
end
 figure;imshow(ReImage,[]);title('MOD denoising')
 figure;imshow(ReImage_Dec,[]);title('Decrypted MOD denoising')
 figure;imshow(ReImage_Enc,[]);title('Encrypted MOD denoising')

%----------------------------SSIM PSNR and MSE
display('MSE index: [imms_MOD , imms_noisy_Dec, imms_noisy , imms_noisy_Enc]')
imms_MOD = immse(uint8(I(:,:,1)),uint8(ReImage));
imms_noisy = immse(uint8(I(:,:,1)),uint8(B(:,:,1)));
imms_noisy_Dec = immse(uint8(I(:,:,1)),uint8(ReImage_Dec));
imms_noisy_Enc = immse(uint8(I(:,:,1)),uint8(ReImage_Enc));
[imms_MOD, imms_noisy_Dec,imms_noisy,imms_noisy_Enc]

display('similarity index: [ssim_MOD , ssim_MOD_Dec , ssim_noisy , ssim_MOD_Enc]')
ssim_MOD=ssim(uint8(I(:,:,1)),uint8(ReImage));
ssim_noisy=ssim(uint8(I(:,:,1)),uint8(B(:,:,1)));
ssim_MOD_Dec=ssim(uint8(I(:,:,1)),uint8(ReImage_Dec));
ssim_MOD_Enc=ssim(uint8(I(:,:,1)),uint8(ReImage_Enc));
[ssim_MOD,ssim_MOD_Dec,ssim_noisy,ssim_MOD_Enc]

display('PSNR index: [psnr_MOD , psnr_MOD_Dec , psnr_noisy , psnr_MOD_Enc]')
psnr_MOD = psnr(uint8(I(:,:,1)),uint8(ReImage));
psnr_noisy = psnr(uint8(I(:,:,1)),uint8(B(:,:,1)));
psnr_MOD_Dec = psnr(uint8(I(:,:,1)),uint8(ReImage_Dec));
psnr_MOD_Enc = psnr(uint8(I(:,:,1)),uint8(ReImage_Enc));
[psnr_MOD,psnr_MOD_Dec,psnr_noisy,psnr_MOD_Enc]
    