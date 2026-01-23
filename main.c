#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <immintrin.h>
#include "bfloat16.h"
#include "png/png.h"
#include <gl/gl.h>
#include <windows.h>

#include "window.c"

#define IN 307200
#define OUT 1024
#define W 640
#define H 480

typedef struct {
 int in_channels;
 int out_channels; //в самом конце, наверно 6 чисел (по две координаты левого верхнего и правого нижнего углов, наклон левой стороны и наклон верхней стороны)
 int kernel_size;
 float *weights;
 float *biases;
}ConvLayer;


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

float * dense_layer(ConvLayer *layer,
 unsigned char* input,float *output,
 int h, int w){

 int H_out=h-layer->kernel_size+1;
 int W_out=w-layer->kernel_size+1;
 for(int y=0;y<H_out;y++){
  for(int x=0;x<W_out;x++){
   for(int oc=0;oc<layer->out_channels;oc++){
    float summ=layer->biases[oc];
    for(int ic=0;ic<layer->in_channels;ic++){
     for(int ky=0;ky<3;ky++){
      for(int kx=0;kx<3;kx++){
       summ+=(float)(input[((y+ky)*W+x+kx)*layer->in_channels+ic]/255.0f)*layer->weights[kx+3*ky];
      }
     }
    }
    output[oc*H_out*W_out+y*W_out+x]=summ;
   }
  }
 }
printf("A\n");
 return output;
}


void conv_backward(ConvLayer *layer,float *input,float *d_output,ConvGradients *grad,int h,int w){
 int H_out=H-2;
 int W_out=W-2;
 memset(grad->d_weights,0,layer->out_channels*layer->in_channels*9*sizeof(float));
 memset(grad->d_biases,0,layer->out_channels*sizeof(float));
 for(int oc=0;oc<layer->out_channels;oc++){
  for(int ic=0;ic<layer->in_channels;ic++){
   for(int ky=0;ky<3;ky++){
    for(int kx=0;kx<3;kx++){
     float grad_sum=0;
     for(int y=0;y<H_out;y++){
      for(int x=0;x<W_out;x++){
       grad_sum+=input[ic*h*w+w*(ky+y)+kx+x]*d_output[oc*H_out*W_out+y*W_out+x];
      }
     }
     grad->d_weights[oc*layer->in_channels*9+ic*9+ky*3+kx]=grad_sum;
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
   dense_layer(layer,data->images[batch],output,H,W);
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
 ConvLayer l1;
 l1.in_channels=3;
 l1.out_channels=32;
 l1.kernel_size=3;
 l1.weights=malloc(sizeof(float)*l1.kernel_size*l1.kernel_size*l1.out_channels);
 l1.biases=malloc(sizeof(float)*l1.out_channels);
 float *result=malloc(H*W*sizeof(float)*l1.out_channels);
printf("%i\n",H*W);
 for(int i=0;i<l1.out_channels;i++)
  l1.biases[i]=0;
 srand(time(0));

 for(int i=0;i<l1.kernel_size*l1.kernel_size*l1.out_channels;i++)
  l1.weights[i]=rand()*1.0f/RAND_MAX;
 l1.weights[0]=1;
 l1.weights[1]=1;
 l1.weights[2]=1;
 l1.weights[3]=0;
 l1.weights[4]=0;
 l1.weights[5]=0;
 l1.weights[6]=1;
 l1.weights[7]=1;
 l1.weights[8]=1;

 png_image data;
 char *img;

 int step;
  memset(&data,0,sizeof(data));
  data.format = PNG_FORMAT_RGB;
  data.version = PNG_IMAGE_VERSION;
  if(png_image_begin_read_from_file(&data,"test.png")!=0){
    img = malloc(PNG_IMAGE_SIZE(data));
    png_image_finish_read(&data, NULL, img, 0, NULL);
    printf("%i %i %i (%i)\n",data.width,data.height,PNG_IMAGE_SIZE(data),data.format);
    if(data.format==0)
      step=1;
    else if(data.format==2)
      step=3;
    png_image_free(&data);
  }
 dense_layer(&l1,img,result,H,W);
 createWindow();
/*
 for(int i=0;i<W*H;i++)
  printf("%.3f ",result[i]);
 printf("\n");*/
}