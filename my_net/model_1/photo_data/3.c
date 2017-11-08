#include<stdio.h>
#include<stdlib.h>

#define x_row 10
#define x_col 784
#define b_row 10
#define b_col 1
void main(){
FILE *fp;
int i,j;
float x[x_row][x_col];
float b[b_row][b_col];
/*
if((fp=fopen("x.txt","r"))==NULL)
 { 
  printf(" can't open");
  exit(0);
 }else{
   for(i=0;i<x_row;i++){
    for(j=0;j<x_col;j++){
     fscanf(fp,"%f",&x[i][j]);
    }
   }
 fclose(fp);
}
for(i=0;i<x_row;i++){
  for(j=0;j<784;j++){
    printf("%f\t",x[i][j]);
  }
  printf("\n");
}*/

 if((fp=fopen("b.txt","r"))==NULL)
 { 
  printf(" can't open");
  exit(0);
 }else{
   for(i=0;i<b_row;i++){
    for(j=0;j<b_col;j++){
     fscanf(fp,"%f",&b[i][j]);
    }
   }
 fclose(fp);
}

for(i=0;i<b_row;i++){
  for(j=0;j<b_col;j++){
    printf("%f\t",b[i][j]);
  }
  printf("\n");
}
}
