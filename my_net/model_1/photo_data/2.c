#include<stdio.h>
int max(float array[],int a);

 int max(float *array,int n){
  int i,index;
  float maxnum;
  for(i=0;i<n;i++){
    if(maxnum<array[i]){
      maxnum=array[i];
      index=i;
    }
  }
  return(index);
 }



 void main(){
 int i,j,k,r1,c1,r2,c2,r3,c3;
 float a[2][2]={{1,2},{1,2}};
 float b[2][2]={{1,2},{1,2}};
 float c[2][2];
/*
 for(i=0;i<sizeof(a)/sizeof(a[0]);i++){
   for(j=0;j<sizeof(b[0])/sizeof(b[0][0]);j++){
     int s=0;
     for(k=0;k<sizeof(b)/sizeof(b[0]);k++){
       s+=a[i][k]*b[k][j];
     }
     c[i][j]=s;
     printf("%f\n",c[i][j]);
   }
 }*/
 printf("%d\n",max(a[0],2));
 
 for(i=0;i<sizeof(a)/sizeof(a[0]);i++){
   for(j=0;j<sizeof(a[0])/sizeof(a[0][0]);j++){
     c[i][j]=a[i][j]+b[i][j];
     printf("%f\t",c[i][j]);
   }
   printf("\n");
 }
}
