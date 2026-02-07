#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <immintrin.h>
#include <windows.h>
#include <gl/gl.h>
#include "png/png.h"


#include "save_png.c"

unsigned char texture[2000000];
#define W 640
#define H 480
int stride=1;
unsigned char *img;
unsigned char *img1;

typedef struct {
 int in_channels;
 int out_channels; //в самом конце, наверно 6 чисел (по две координаты левого верхнего и правого нижнего углов, наклон левой стороны и наклон верхней стороны)
 int kernel_size;
 float *weights;
 float *biases;
}ConvLayer;

ConvLayer l1;
ConvLayer l2;
float *result;

#include "window.c"

typedef struct{
 float* d_weights;  // Gradients for weights
 float* d_biases;   // Gradients for biases
 float* d_input;    // Gradients to pass backward
}ConvGradients;

typedef struct{
 float **images;
 float **targets;
 int num_batches;
}TrainingData;

typedef struct{
 float *weights;
 float bias;
 int out_channels;
 int in_channels;
}FCLayer;

void fully_conneted_layer(float *input,float *output,FCLayer *layer,int w,int h){
 for(int oc=0;oc<layer->out_channels;oc++){
  float summ=layer->bias;
  for(int j=0;j<h;j++){
   for(int i=0;i<w;i++){
    summ+=input[i+j*w]*layer->weights[i+j*w];
   }
  }
  output[oc]=summ;
 }
}


float * dense_layer_strided(ConvLayer *layer,
 float* input,float *output,
 int h, int w,int stride){

 int H_out=h/stride-layer->kernel_size+1;
 int W_out=w/stride-layer->kernel_size+1;
 for(int y=0;y<H_out;y++){

  for(int x=0;x<W_out;x++){
   for(int oc=0;oc<layer->out_channels;oc++){
    float summ=layer->biases[oc];
    for(int ic=0;ic<layer->in_channels;ic++){
     for(int ky=0;ky<layer->kernel_size;ky++){
      for(int kx=0;kx<layer->kernel_size;kx++){
       size_t input_idx = ((y+ky)*W_out*stride+(x+kx))*layer->in_channels*stride+ic*stride;
       size_t weight_idx=kx+layer->kernel_size*ky+ic*layer->kernel_size*layer->kernel_size+oc*layer->kernel_size*layer->kernel_size*layer->in_channels;
//9830400
       summ+=input[input_idx]*layer->weights[weight_idx];

      }
     }
    }

    output[layer->out_channels*(y*W_out+x)+oc]=summ;

   }
  }
//  printf("[%.1f%%]\n",y*100.0/H_out);
 }
 printf("[100.0%%]\n");
 return output;
}

void conv_backward(ConvLayer *layer,float *input,float *d_output,ConvGradients *grad,int h,int w){
 int H_out=H-2;
 int W_out=W-2;
 memset(grad->d_weights,0,layer->out_channels*layer->in_channels*9*sizeof(float));
 memset(grad->d_biases,0,layer->out_channels*sizeof(float));
 for(int i=0;i<640*480*3;i++)
  if(d_output[i]!=0)
   printf("d_output[%i]=%f\n",i,d_output[i]);

 for(int oc=0;oc<layer->out_channels;oc++){
  for(int ic=0;ic<layer->in_channels;ic++){
   for(int ky=0;ky<3;ky++){
    for(int kx=0;kx<3;kx++){
     float grad_sum=0;
     for(int y=0;y<H_out;y++){
      for(int x=0;x<W_out;x++){
       size_t input_idx=ic+(x+kx)*layer->in_channels+(y+ky)*w*layer->in_channels;
       size_t d_output_idx=oc+x*layer->out_channels+y*W_out*layer->out_channels;
       grad_sum+=input[input_idx]*d_output[d_output_idx];
      }
     }
     grad->d_weights[oc*layer->in_channels*9+ic*9+ky*3+kx]=grad_sum;
     if(grad_sum)
      printf("grad_sum[%i][%i][%i][%i]=%f\n",ic,oc,kx,ky,grad_sum);
    }
   }
  }
  float grad_bias=0;
  for(int i=0;i<H_out*W_out;i++)
   grad_bias+=d_output[oc*W_out*H_out+i];
  grad->d_biases[oc]=grad_bias;
 }
}

void conv_update(ConvLayer *layer,ConvGradients *grad,float learning_rate){
 int total_weights=layer->out_channels*layer->in_channels*9;
 for(int i=0;i<total_weights;i++){
  layer->weights[i]-=learning_rate*grad->d_weights[i];
 }
 for(int i=0;i<layer->out_channels;i++){
  layer->biases[i]-=learning_rate*grad->d_biases[i];
 }
}

float compute_loss(float *output,float *targets){
 return 1.0f;
}

float *compute_loss_gradients(float *output,float *target){
 float *summ=malloc(sizeof(float)*6);
 for(int i=0;i<6;i++){
  summ[i]=target[i]-output[i];
  
 }
 return summ;
}
void train_conv_layer(ConvLayer *layer,TrainingData *data,int epochs,float learning_rate){
 ConvGradients grad;
 grad.d_weights=malloc(sizeof(float)*layer->kernel_size*layer->kernel_size);
 grad.d_biases=malloc(sizeof(float)*layer->out_channels);
 for(int epoch=0;epoch<epochs;epoch++){
  float total_loss=0;
  for(int batch=0;batch<data->num_batches;batch++){
   float *output=malloc((W+1)*(H*1)*sizeof(float));
   dense_layer_strided(layer,data->images[batch],output,H,W,stride);
   float loss=compute_loss(output,data->targets[batch]);
   total_loss+=loss;
   float *d_output=compute_loss_gradients(output,data->targets[batch]);
   conv_backward(layer,data->images[batch],d_output,&grad,480,640);
   conv_update(layer,&grad,learning_rate);
  }
  printf("Epoch %d, Loss: %f\n",epoch,total_loss/data->num_batches);
 }
}

int main(){
 l1.in_channels=3;
 l1.out_channels=32;
 l1.kernel_size=3;
 l1.weights=malloc(sizeof(float)*l1.kernel_size*l1.kernel_size*l1.out_channels*l1.in_channels);
 l1.biases=malloc(sizeof(float)*l1.out_channels);



 result=malloc(H*W*sizeof(float)*l1.out_channels);
 for(int i=0;i<l1.out_channels;i++)
  l1.biases[i]=0;
 srand(time(0));

 for(size_t i=0;i<l1.kernel_size*l1.kernel_size*l1.out_channels*l1.in_channels;i++){
  l1.weights[i]=rand()*1.0f/RAND_MAX;
 }
printf("[0]=%f [1]=%f [2]=%f\n",l1.weights[0],l1.weights[1],l1.weights[2]);

 //нормализация весов

 for(size_t j=0;j<l1.in_channels*l1.out_channels;j++){
  float pixel_weight_summ=0;
  for(size_t i=0;i<l1.kernel_size*l1.kernel_size;i++){
   pixel_weight_summ+=l1.weights[i+j*l1.kernel_size*l1.kernel_size];
  }
  for(size_t i=0;i<l1.kernel_size*l1.kernel_size;i++){
   if(pixel_weight_summ!=0)
    l1.weights[i+j*l1.kernel_size*l1.kernel_size]/=pixel_weight_summ*0.5;
  }
 }

 l2.in_channels=l1.out_channels;
 l2.out_channels=3;
 l2.kernel_size=3;
 l2.weights=malloc(sizeof(float)*l2.kernel_size*l2.kernel_size*l2.out_channels*l2.in_channels);
 l2.biases=malloc(sizeof(float)*l2.out_channels);

 for(size_t i=0;i<l2.kernel_size*l2.kernel_size*l2.out_channels*l2.in_channels;i++){
  l2.weights[i]=rand()*1.0f/RAND_MAX;
 }

 for(size_t j=0;j<l2.in_channels*l2.out_channels;j++){
  float pixel_weight_summ=0;
  for(size_t i=0;i<l2.kernel_size*l2.kernel_size;i++){
   pixel_weight_summ+=l2.weights[i+j*l2.kernel_size*l2.kernel_size];
  }
  for(size_t i=0;i<l2.kernel_size*l2.kernel_size;i++){
   if(pixel_weight_summ!=0)
    l2.weights[i+j*l2.kernel_size*l2.kernel_size]/=pixel_weight_summ;
  }
 }


printf("[0]=%f [1]=%f [2]=%f\n",l1.weights[0],l1.weights[1],l1.weights[2]);
 float *result2=malloc(H*W*sizeof(float)*l2.out_channels);

 float *image;
 png_image data;
 int step;
 unsigned int image_size;
 memset(&data,0,sizeof(data));
 data.format = PNG_FORMAT_RGB;
 data.version = PNG_IMAGE_VERSION;
 if(png_image_begin_read_from_file(&data,"test.png")!=0){
   img = malloc(PNG_IMAGE_SIZE(data));
   image_size=PNG_IMAGE_SIZE(data);
   image=malloc(sizeof(float)*image_size);
   png_image_finish_read(&data, NULL, img, 0, NULL);
   printf("%i %i %i (%i)\n",data.width,data.height,PNG_IMAGE_SIZE(data),data.format);
   if(data.format==0)
     step=1;
   else if(data.format==2){
     step=3;
    for(size_t i=0;i<image_size;i++)
     image[i]=img[i]/255.0f;
   }else if(data.format==3){
    step=4;
    for(size_t i=0;i<image_size/4;i++){
     image[i*3]=img[i*4]*1.0f/255.0f;
     image[i*3+1]=img[i*4+1]*1.0f/255.0f;
     image[i*3+2]=img[i*4+2]*1.0f/255.0f;
    }
   }
   png_image_free(&data);
 }

 float *image1;
 memset(&data,0,sizeof(data));
 data.format = PNG_FORMAT_RGB;
 data.version = PNG_IMAGE_VERSION;
 if(png_image_begin_read_from_file(&data,"test-result.png")!=0){
   img1 = malloc(PNG_IMAGE_SIZE(data));
   image_size=PNG_IMAGE_SIZE(data);
   image1=malloc(sizeof(float)*image_size);
   png_image_finish_read(&data, NULL, img1, 0, NULL);
   printf("%i %i %i (%i)\n",data.width,data.height,PNG_IMAGE_SIZE(data),data.format);
   if(data.format==0)
     step=1;
   else if(data.format==2){
     step=3;
    for(int i=0;i<image_size;i++){
     image1[i]=img1[i]/255.0f;
    }
   }else if(data.format==3){
    step=4;
    for(int i=0;i<image_size/4;i++){
     image1[i*3]=img1[i*4]*1.0f/255.0f;
     image1[i*3+1]=img1[i*4+1]*1.0f/255.0f;
     image1[i*3+2]=img1[i*4+2]*1.0f/255.0f;
    }
   }
   png_image_free(&data);
 }else{
  return -1;
 }

for(size_t i=0;i<image_size;i++)
 image1[i]=0;
/*
for(size_t i=0;i<image_size;i++){
 if(image1[i]!=0)
  printf("image2 [%i]=%f\n",i,image1[i]);
}
*/
 dense_layer_strided(&l1,image,result,H,W,stride);
 dense_layer_strided(&l2,result,result2,H,W,stride);

 ConvGradients grad;

 grad.d_weights=malloc(sizeof(float)*l1.kernel_size*l1.kernel_size*l1.in_channels*l1.out_channels);
 grad.d_biases=malloc(sizeof(float)*l1.out_channels);
// grad.d_input=malloc(sizeof(float)*l1.kernel_size*l1.in_channels);
/*
 conv_backward(&l1,image,image1,&grad,H,W);

printf("before: w5=%.5f\n",l1.weights[5]);
 float learning_rate=0.001;

 conv_update(&l1,&grad,learning_rate);
printf("after: w5=%.5f\n",l1.weights[5]);

 dense_layer_strided(&l1,image,result,H,W,stride);
printf("B:result[0]=%f\n",result[0]);
*/

 for(int i=0;i<640*480;i++){
//  texture[i*3]=(unsigned char)(image1[i*l1.out_channels]*255);

  texture[i*3]=(unsigned char)(result[i*l1.out_channels]*255);
  texture[i*3+1]=(unsigned char)(result[i*l1.out_channels+1]*255);
  texture[i*3+2]=(unsigned char)(result[i*l1.out_channels+2]*255);
/*
  texture[i*3]=(unsigned char)(result2[i*l2.out_channels]*255);
  texture[i*3+1]=(unsigned char)(result2[i*l2.out_channels+1]*255);
  texture[i*3+2]=(unsigned char)(result2[i*l2.out_channels+2]*255);
*/
 }
 unsigned char *FUCK=malloc(W*3);
 for(int y=0;y<H;y++){
  memcpy(FUCK,texture+y*W*3,W*3);
 }
 printf("ASD\n");
 save_png("layer1.png",texture,W,H);
 createWindow();
}
