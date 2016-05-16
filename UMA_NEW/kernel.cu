#include <cuda.h>
#include <cuda_runtime.h>

#include "Snapshot.h"
#include "Agent.h"

static bool *Gdir=NULL,*dev_dir;
static double *Gweights,*dev_weights,*Gthresholds,*dev_thresholds;
static bool *Gobserve,*dev_observe;
static bool *Gdfs,*dev_dfs;
static bool *Gsignal,*dev_signal,*Gload,*dev_load;
static bool *Gmask,*dev_mask;//bool value for mask signal in halucinate
static int *dev_actionlist;//action list in halucinate
static int *tmp_signal,*tmp_load;//tmp variable for dfs on GPU
static bool *dfs_flag,*dev_dfs_flag;

//helper function
__host__ __device__ int compi_GPU(int x){
	if(x%2==0) return x+1;
	else return x-1;
}

__host__ __device__ int ind(int row,int col,int width){
	return row*width+col;
}

__global__ void conjunction_kernel(bool *b1,bool *b2,int size){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<size){
		b1[index]=b1[index]&&b2[index];
	}
}

__global__ void disjunction_kernel(bool *b1,bool *b2,int size){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<size){
		b1[index]=b1[index]||b2[index];
	}
}


__global__ void negate_disjunction_star_kernel(bool *b1,bool *b2,int size){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<size){
		if(index%2==0){
			b1[index]=b1[index]&&!b2[index+1];
		}
		else{
			b1[index]=b1[index]&&!b2[index-1];
		}
	}
}

__global__ void int2bool_kernel(bool *b,int *i,int size){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<size){
		if(i[index]==1) b[index]=true;
		else b[index]=false;
	}
}

__global__ void bool2int_kernel(int *i,bool *b,int size){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<size){
		if(b[index]) i[index]=0;
		else i[index]=-1;
	}
}

//helper function

__device__ bool implies_GPU(int row,int col,int width,double *weights,double threshold){
	double rc=weights[ind(row,col,width)];
	double r_c=weights[ind(compi_GPU(row),col,width)];
	double rc_=weights[ind(row,compi_GPU(col),width)];
	double r_c_=weights[ind(compi_GPU(row),compi_GPU(col),width)];
	double epsilon=(rc+r_c+rc_+r_c_)*threshold;
	double m=min(epsilon,min(rc,min(r_c,r_c_)));
	return rc_<m;
}

__device__ bool equivalent_GPU(int row,int col,int width,double *weights,double threshold){
	double rc=weights[ind(row,col,width)];
	double r_c=weights[ind(compi_GPU(row),col,width)];
	double rc_=weights[ind(row,compi_GPU(col),width)];
	double r_c_=weights[ind(compi_GPU(row),compi_GPU(col),width)];
	double epsilon=(rc+r_c+rc_+r_c_)*threshold;
	return rc_==0&&r_c==0;
}

__device__ void orient_square_GPU(bool *dir,double *weights,double *thresholds,int x,int y,int width){
	dir[ind(x,y,width)]=false;
	dir[ind(x,compi_GPU(y),width)]=false;
	dir[ind(compi_GPU(x),y,width)]=false;
	dir[ind(compi_GPU(x),compi_GPU(y),width)]=false;
	dir[ind(y,x,width)]=false;
	dir[ind(compi_GPU(y),x,width)]=false;
	dir[ind(y,compi_GPU(x),width)]=false;
	dir[ind(compi_GPU(y),compi_GPU(x),width)]=false;

	int square_is_oriented=0;
	for(int i=0;i<2;++i){
		for(int j=0;j<2;++j){
			int sx=x+i;
            int sy=y+j;
			if(square_is_oriented==0){
				if(implies_GPU(sy,sx,width,weights,thresholds[ind(sy,sx,width)])){
					dir[ind(sy,sx,width)]=true;
					dir[ind(compi_GPU(sx),compi_GPU(sy),width)]=true;
					dir[ind(sx,sy,width)]=false;
                    dir[ind(compi_GPU(sy),compi_GPU(sx),width)]=false;
                    dir[ind(sx,compi_GPU(sy),width)]=false;
                    dir[ind(compi_GPU(sy),sx,width)]=false;
                    dir[ind(sy,compi_GPU(sx),width)]=false;
                    dir[ind(compi_GPU(sx),sy,width)]=false;
                    square_is_oriented=1;
				}//implies
				if(equivalent_GPU(sy,sx,width,weights,thresholds[ind(sy,sx,width)])){
					dir[ind(sy,sx,width)]=true;
					dir[ind(sx,sy,width)]=true;
					dir[ind(compi_GPU(sx),compi_GPU(sy),width)]=true;
                    dir[ind(compi_GPU(sy),compi_GPU(sx),width)]=true;
					dir[ind(sx,compi_GPU(sy),width)]=false;
                    dir[ind(compi_GPU(sy),sx,width)]=false;
                    dir[ind(sy,compi_GPU(sx),width)]=false;
                    dir[ind(compi_GPU(sx),sy,width)]=false;
                    square_is_oriented=1;
				}//equivalent
			}//square_is_oriented
		}//j
	}//i
}

__global__ void update_weights_kernel(double *weights,bool *observe,int size){
	int indexX=blockDim.x*blockIdx.x+threadIdx.x;
	int indexY=blockDim.y*blockIdx.y+threadIdx.y;
	if(indexX<size&&indexY<size){
		weights[ind(indexY,indexX,size)]+=observe[indexX]*observe[indexY];
	}
}

__global__ void orient_all_kernel(bool *dir,double *weights,double *thresholds,int size){
	int indexX=blockDim.x*blockIdx.x+threadIdx.x;
	int indexY=blockDim.y*blockIdx.y+threadIdx.y;
	/*if(indexX<size){//possible optimazation
		if(indexY>indexX) orient_square_GPU(dir,weights,thresholds,2*(size/2-1-indexX),2*(size/2-1-indexY),size);
		else if(indexY<indexX) orient_square_GPU(dir,weights,thresholds,2*indexX,2*indexY,size);
	}*/
	if(indexX<size/2&&indexY<indexX){
		orient_square_GPU(dir,weights,thresholds,indexX*2,indexY*2,size);
	}
}

__global__ void dfs_GPU(bool *dir,int *dfs,bool *flag,int size){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<size&&dfs[index]==0){
		for(int j=0;j<size;++j){
			if(j==index) continue;
			if(dir[ind(index,j,size)]){
				atomicMax(dfs+j,0);
				flag[0]=true;
			}
		}
		atomicMax(dfs+index,1);
	}
}

//mask=Signal([(ind in actions_list) for ind in xrange(self._SIZE)])
__global__ void mask_kernel(bool *mask,int *actionlist,int size){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<size){
		for(int i=0;i<size;++i){
			if(index==actionlist[i]){
				mask[index]=true;
				return;
			}
		}
		mask[index]=false;
	}
}

//before invoke this function make sure dev_load and dev_signal have correct data
//the computed data will be in dev_load
void Snapshot::propagate_GPU(){
	bool2int_kernel<<<(size+255)/256,256>>>(tmp_load,dev_load,size);
	bool2int_kernel<<<(size+255)/256,256>>>(tmp_signal,dev_signal,size);

	dfs_flag[0]=true;
	while(dfs_flag[0]){
		cudaMemset(dev_dfs_flag,false,sizeof(bool));
		dfs_GPU<<<(size+255)/256,256>>>(dev_dir,tmp_load,dev_dfs_flag,size);
		cudaMemcpy(dfs_flag,dev_dfs_flag,sizeof(bool),cudaMemcpyDeviceToHost);
	}
	int2bool_kernel<<<(size+255)/256,256>>>(dev_load,tmp_load,size);
	//load=self.up(load)
	dfs_flag[0]=true;
	while(dfs_flag[0]){
		cudaMemset(dev_dfs_flag,false,sizeof(bool));
		dfs_GPU<<<(size+255)/256,256>>>(dev_dir,tmp_signal,dev_dfs_flag,size);
		cudaMemcpy(dfs_flag,dev_dfs_flag,sizeof(bool),cudaMemcpyDeviceToHost);
	}
	int2bool_kernel<<<(size+255)/256,256>>>(dev_signal,tmp_signal,size);
	//mask_pos=self.up(signal)
	
	disjunction_kernel<<<(size+255)/256,256>>>(dev_load,dev_signal,size);
	negate_disjunction_star_kernel<<<(size+255)/256,256>>>(dev_load,dev_signal,size);
	
	cudaMemcpy(Gload,dev_load,size*sizeof(bool),cudaMemcpyDeviceToHost);
}

void Snapshot::setSignal(vector<bool> observe){
	for(int i=0;i<observe.size();++i){
		Gobserve[i]=observe[i];
	}
	cudaMemcpy(dev_observe,Gobserve,size*sizeof(bool),cudaMemcpyHostToDevice);
}

void Snapshot::update_state_GPU(bool mode){//true for decide
	dim3 dimGrid((size+15)/16,(size+15)/16);
	dim3 dimBlock(16,16);
	update_weights_kernel<<<dimGrid,dimBlock>>>(dev_weights,dev_observe,size);
	//update_weight

	if(mode){
		dim3 dimGrid1((size/2+15)/16,(size/2+15)/16);
		dim3 dimBlock1(16,16);
		orient_all_kernel<<<dimGrid1,dimBlock1>>>(dev_dir,dev_weights,dev_thresholds,size);
	}//orient_all

	cudaMemcpy(dev_signal,dev_observe,size*sizeof(bool),cudaMemcpyDeviceToDevice);
	cudaMemset(dev_load,false,size*sizeof(bool));
	propagate_GPU();
}

void Snapshot::halucinate_GPU(vector<int> actions_list){
	//mask=Signal([(ind in actions_list) for ind in xrange(self._SIZE)])
	vector<bool> mask=initMask(actions_list);
	vector<int> v;
	for(int i=0;i<actions_list.size();++i){
		for(int j=0;j<size;++j){
			if(context.find(pair<int,int>(i,j))!=context.end()&&Gload[j]){
				v.push_back(context[pair<int,int>(i,j)]);
			}
		}
	}
	//relevant_pairs=[(act,ind) for act in actions_list for ind in xrange(self._SIZE) if (act,ind) in self._CONTEXT and self._CURRENT.value(ind)]
	//map(mask.set,[self._CONTEXT[i,j] for i,j in relevant_pairs],[True for i,j in relevant_pairs])
	for(int i=0;i<v.size();++i) mask[v[i]]=true;
	
	for(int i=0;i<mask.size();++i){
		Gmask[i]=mask[i];
	}
	cudaMemcpy(dev_mask,Gmask,size*sizeof(bool),cudaMemcpyHostToDevice);
	//copy data
	cudaMemcpy(dev_signal,dev_mask,size*sizeof(bool),cudaMemcpyDeviceToDevice);
	//dev_load already there
	propagate_GPU();
	//return self.propagate(mask,self._CURRENT)
}

void Snapshot::initData(int size,double threshold,vector<vector<int>> context_key,vector<int> context_value){
	this->size=size;
	this->threshold=threshold;
	Gdir=new bool[size*size];
	Gweights=new double[size*size];
	Gthresholds=new double[size*size];
	Gobserve=new bool[size];
	Gdfs=new bool[1];
	Gsignal=new bool[size];
	Gload=new bool[size];
	tmp_signal=new int[size];
	tmp_load=new int[size];
	dfs_flag=new bool[1];
	Gmask=new bool[size];
	
	cudaMalloc(&dev_dir,size*size*sizeof(bool));
	cudaMalloc(&dev_thresholds,size*size*sizeof(double));
	cudaMalloc(&dev_weights,size*size*sizeof(double));
	cudaMalloc(&dev_observe,size*sizeof(bool));
	cudaMalloc(&dev_dfs,sizeof(bool));
	cudaMalloc(&dev_signal,size*sizeof(bool));
	cudaMalloc(&dev_load,size*sizeof(bool));
	cudaMalloc(&tmp_signal,size*sizeof(int));
	cudaMalloc(&tmp_load,size*sizeof(int));
	cudaMalloc(&dev_dfs_flag,sizeof(bool));

	cudaMalloc(&dev_mask,size*sizeof(bool));

	for(int i=0;i<size;++i){
		for(int j=0;j<size;++j){
			Gthresholds[i*size+j]=threshold;
		}
	}
	//init threshold

	for(int i=0;i<context_key.size();++i){
		context[pair<int,int>(context_key[i][0],context_key[i][1])]=context_value[i];
	}
	cout<<"succeed"<<endl;
}

vector<bool> Snapshot::getLoad(){
	vector<bool> result;
	for(int i=0;i<size;++i){
		result.push_back(Gload[i]);
	}
	return result;
}