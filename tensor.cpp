#include "tensor.h"
#include <iostream>
#include <functional>
#include <algorithm>
#include <numeric>
#include <random>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<dtype> dis(-1,1);

dtype randVal()
{
	return dis(gen);
}

//TensorPtr Definition
TensorPtr::TensorPtr(dtype scalar,const std::vector<std::shared_ptr<TensorPtr>> & parents,std::function<void(const std::vector<dtype> &)> gradfn): _data{scalar}, _shape{}, _stride{}, _parents(parents), _gradfn(gradfn)
{
	zero_grad();
}

TensorPtr::TensorPtr(const std::vector<dtype> &vec,const std::vector<std::shared_ptr<TensorPtr>> & parents,std::function<void(const std::vector<dtype> &)> gradfn): _data(vec), _shape{vec.size()}, _stride{1}, _parents(parents), _gradfn(gradfn)
{
	zero_grad();
}

TensorPtr::TensorPtr(const std::vector<std::vector<dtype>> &vec,const std::vector<std::shared_ptr<TensorPtr>> & parents,std::function<void(const std::vector<dtype> &)> gradfn): _shape{vec.size(),vec[0].size()}, _stride{vec[0].size(),1}, _parents(parents), _gradfn(gradfn)
{
	for(size_t i=1;i<vec.size();i++)
	{
		if(vec[i].size()!=vec[0].size()) throw std::runtime_error("Inconsistent tensor(Expected length "+std::to_string(vec[0].size())+" but got length "+std::to_string(vec[i].size())+" at index "+std::to_string(i)+")!");
	}
	_data.reserve(vec.size()*vec[0].size());
	for(size_t i=0;i<vec.size();i++)
	{
		for(size_t j=0;j<vec[0].size();j++)
		{
			_data.push_back(vec[i][j]);
		}
	}
	zero_grad();
}

std::shared_ptr<TensorPtr> TensorPtr::add(const std::shared_ptr<TensorPtr> &a,const std::shared_ptr<TensorPtr> &b)
{	
	//scalar + scalar
	if(a->_shape.size()==0 && b->_shape.size()==0)
	{
		dtype result=a->_data[0]+b->_data[0];
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			a->_grad[0]+=grad_out[0];
			b->_grad[0]+=grad_out[0];
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		return ptr;
	}
	
	//scalar + 1D
	else if(a->_shape.size()==0 && b->_shape.size()==1)
	{
		std::vector<dtype> result;
		result.reserve(b->_shape[0]);
		for(size_t i=0;i<b->_shape[0];i++)
		{
			result.push_back(a->_data[0]+b->_data[i]);
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			a->_grad[0]+=std::accumulate(grad_out.begin(),grad_out.end(),dtype(0));
			std::transform(b->_grad.begin(),b->_grad.end(),grad_out.begin(),b->_grad.begin(),std::plus<dtype>());
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		return ptr;
	}
	//1D + scalar
	else if(a->_shape.size()==1 && b->_shape.size()==0)
	{
		std::vector<dtype> result;
		result.reserve(a->_shape[0]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			result.push_back(a->_data[i]+b->_data[0]);
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			std::transform(a->_grad.begin(),a->_grad.end(),grad_out.begin(),a->_grad.begin(),std::plus<dtype>());
			b->_grad[0]+=std::accumulate(grad_out.begin(),grad_out.end(),dtype(0));
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		return ptr;
	}
	//1D+1D
	else if(a->_shape.size()==1 && b->_shape.size()==1)
	{
		if(a->_shape!=b->_shape) throw std::runtime_error("Both 1D tensors must be of same shape for addition!");
		std::vector<dtype> result;
		result.reserve(a->_shape[0]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			result.push_back(a->_data[i]+b->_data[i]);
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			std::transform(a->_grad.begin(),a->_grad.end(),grad_out.begin(),a->_grad.begin(),std::plus<dtype>());
			std::transform(b->_grad.begin(),b->_grad.end(),grad_out.begin(),b->_grad.begin(),std::plus<dtype>());
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		return ptr;
	}
	//scalar + 2D
	else if(a->_shape.size()==0 && b->_shape.size()==2)
	{
		std::vector<dtype> result;
		result.reserve(b->_shape[0] * b->_shape[1]);
		for(size_t i=0;i<b->_shape[0];i++)
		{
			for(size_t j=0;j<b->_shape[1];j++)
			{
				result.push_back(a->_data[0]+b->_data[i * b->_stride[0] + j * b->_stride[1]]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			a->_grad[0]+=std::accumulate(grad_out.begin(),grad_out.end(),dtype(0));
			std::transform(b->_grad.begin(),b->_grad.end(),grad_out.begin(),b->_grad.begin(),std::plus<dtype>());
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{b->_shape[0],b->_shape[1]};
		ptr->_stride=std::vector<size_t>{b->_shape[1],1};
		return ptr;
	}
	//2D+scalar
	else if(a->_shape.size()==2 && b->_shape.size()==0)
	{
		std::vector<dtype> result;
		result.reserve(a->_shape[0] * a->_shape[1]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			for(size_t j=0;j<a->_shape[1];j++)
			{
				result.push_back(a->_data[i * a->_stride[0] + j * a->_stride[1]] + b->_data[0]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			std::transform(a->_grad.begin(),a->_grad.end(),grad_out.begin(),a->_grad.begin(),std::plus<dtype>());
			b->_grad[0] += std::accumulate(grad_out.begin(),grad_out.end(),dtype(0));
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{a->_shape[0],a->_shape[1]};
		ptr->_stride=std::vector<size_t>{a->_shape[1],1};
		return ptr;
	}
	//1D+2D
	else if(a->_shape.size()==1 && b->_shape.size()==2)
	{
		if(a->_shape[0]!=b->_shape[1]) throw std::runtime_error("Shape[0] of 1D tensor must be equal to Shape[1] of 2D tensor for addition!");
		std::vector<dtype> result;
		result.reserve(b->_shape[0] * b->_shape[1]);
		for(size_t i=0;i<b->_shape[0];i++)
		{
			for(size_t j=0;j<b->_shape[1];j++)
			{
				result.push_back(a->_data[j] + b->_data[i * b->_stride[0] + j * b->_stride[1]]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			for(size_t i=0;i<b->_shape[0];i++)
			{
				for(size_t j=0;j<b->_shape[1];j++)
				{
					a->_grad[j]+=grad_out[i * b->_stride[0] + j * b->_stride[1]];
					b->_grad[i * b->_stride[0] +j * b->_stride[1]] += grad_out[i * b->_stride[0] + j * b->_stride[1]];
				}
			}
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{b->_shape[0],b->_shape[1]};
		ptr->_stride=std::vector<size_t>{b->_shape[1],1};
		return ptr;
	}
	//2D+1D
	else if(a->_shape.size()==2 && b->_shape.size()==1)
	{
		if(a->_shape[1]!=b->_shape[0]) throw std::runtime_error("Shape[1] of 2D tensor must be equal to Shape[0] of 1D tensor for addition!");
		std::vector<dtype> result;
		result.reserve(a->_shape[0] * a->_shape[1]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			for(size_t j=0;j<a->_shape[1];j++)
			{
				result.push_back(a->_data[i * a->_stride[0] + j * a->_stride[1]] + b->_data[j]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			for(size_t i=0;i<a->_shape[0];i++)
			{
				for(size_t j=0;j<a->_shape[1];j++)
				{
					a->_grad[i * a->_stride[0] +j * a->_stride[1]] += grad_out[i * a->_stride[0] + j * a->_stride[1]];
					b->_grad[j]+=grad_out[i * a->_stride[0] + j * a->_stride[1]];
				}
			}
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{a->_shape[0],a->_shape[1]};
		ptr->_stride=std::vector<size_t>{a->_shape[1],1};
		return ptr;
	}
	//2D+2D
	else if(a->_shape.size()==2 && b->_shape.size()==2)
	{
		if(a->_shape!=b->_shape) throw std::runtime_error("Both 2D tensors must be of same shape for addition!");
		std::vector<dtype> result;
		result.reserve(a->_shape[0] * a->_shape[1]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			for(size_t j=0;j<a->_shape[1];j++)
			{
				result.push_back(a->_data[i * a->_stride[0] + j * a->_stride[1]] + b->_data[i * b->_stride[0] + j * b->_stride[1]]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			for(size_t i=0;i<a->_shape[0];i++)
			{
				for(size_t j=0;j<a->_shape[1];j++)
				{
					a->_grad[i * a->_stride[0] +j * a->_stride[1]] += grad_out[i * a->_stride[0] + j * a->_stride[1]];
					b->_grad[i * b->_stride[0] +j * b->_stride[1]] += grad_out[i * b->_stride[0] + j * b->_stride[1]];
				}
			}
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{b->_shape[0],b->_shape[1]};
		ptr->_stride=std::vector<size_t>{b->_shape[1],1};
		return ptr;
	}
	else
	{
		throw std::runtime_error("Not supported yet!");
	}

}
std::shared_ptr<TensorPtr> TensorPtr::sub(const std::shared_ptr<TensorPtr> &a,const std::shared_ptr<TensorPtr> &b)
{	
	//scalar - scalar
	if(a->_shape.size()==0 && b->_shape.size()==0)
	{
		dtype result=a->_data[0]-b->_data[0];
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			a->_grad[0]+=grad_out[0];
			b->_grad[0]-=grad_out[0];
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		return ptr;
	}
	
	//scalar - 1D
	else if(a->_shape.size()==0 && b->_shape.size()==1)
	{
		std::vector<dtype> result;
		result.reserve(b->_shape[0]);
		for(size_t i=0;i<b->_shape[0];i++)
		{
			result.push_back(a->_data[0]-b->_data[i]);
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			a->_grad[0]+=std::accumulate(grad_out.begin(),grad_out.end(),dtype(0));
			std::transform(b->_grad.begin(),b->_grad.end(),grad_out.begin(),b->_grad.begin(),std::minus<dtype>());
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		return ptr;
	}
	//1D - scalar
	else if(a->_shape.size()==1 && b->_shape.size()==0)
	{
		std::vector<dtype> result;
		result.reserve(a->_shape[0]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			result.push_back(a->_data[i]-b->_data[0]);
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			std::transform(a->_grad.begin(),a->_grad.end(),grad_out.begin(),a->_grad.begin(),std::plus<dtype>());
			b->_grad[0]-=std::accumulate(grad_out.begin(),grad_out.end(),dtype(0));
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		return ptr;
	}
	//1D-1D
	else if(a->_shape.size()==1 && b->_shape.size()==1)
	{
		if(a->_shape!=b->_shape) throw std::runtime_error("Both 1D tensors must be of same shape for subtraction!");
		std::vector<dtype> result;
		result.reserve(a->_shape[0]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			result.push_back(a->_data[i]-b->_data[i]);
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			std::transform(a->_grad.begin(),a->_grad.end(),grad_out.begin(),a->_grad.begin(),std::plus<dtype>());
			std::transform(b->_grad.begin(),b->_grad.end(),grad_out.begin(),b->_grad.begin(),std::minus<dtype>());
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		return ptr;
	}
	//scalar - 2D
	else if(a->_shape.size()==0 && b->_shape.size()==2)
	{
		std::vector<dtype> result;
		result.reserve(b->_shape[0] * b->_shape[1]);
		for(size_t i=0;i<b->_shape[0];i++)
		{
			for(size_t j=0;j<b->_shape[1];j++)
			{
				result.push_back(a->_data[0]-b->_data[i * b->_stride[0] + j * b->_stride[1]]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			a->_grad[0]+=std::accumulate(grad_out.begin(),grad_out.end(),dtype(0));
			std::transform(b->_grad.begin(),b->_grad.end(),grad_out.begin(),b->_grad.begin(),std::minus<dtype>());
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{b->_shape[0],b->_shape[1]};
		ptr->_stride=std::vector<size_t>{b->_shape[1],1};
		return ptr;
	}
	//2D-scalar
	else if(a->_shape.size()==2 && b->_shape.size()==0)
	{
		std::vector<dtype> result;
		result.reserve(a->_shape[0] * a->_shape[1]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			for(size_t j=0;j<a->_shape[1];j++)
			{
				result.push_back(a->_data[i * a->_stride[0] + j * a->_stride[1]] - b->_data[0]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			std::transform(a->_grad.begin(),a->_grad.end(),grad_out.begin(),a->_grad.begin(),std::plus<dtype>());
			b->_grad[0] -= std::accumulate(grad_out.begin(),grad_out.end(),dtype(0));
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{a->_shape[0],a->_shape[1]};
		ptr->_stride=std::vector<size_t>{a->_shape[1],1};
		return ptr;
	}
	//1D-2D
	else if(a->_shape.size()==1 && b->_shape.size()==2)
	{
		if(a->_shape[0]!=b->_shape[1]) throw std::runtime_error("Shape[0] of 1D tensor must be equal to Shape[1] of 2D tensor for addition!");
		std::vector<dtype> result;
		result.reserve(b->_shape[0] * b->_shape[1]);
		for(size_t i=0;i<b->_shape[0];i++)
		{
			for(size_t j=0;j<b->_shape[1];j++)
			{
				result.push_back(a->_data[j] - b->_data[i * b->_stride[0] + j * b->_stride[1]]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			for(size_t i=0;i<b->_shape[0];i++)
			{
				for(size_t j=0;j<b->_shape[1];j++)
				{
					a->_grad[j]+=grad_out[i * b->_stride[0] + j * b->_stride[1]];
					b->_grad[i * b->_stride[0] +j * b->_stride[1]] -= grad_out[i * b->_stride[0] + j * b->_stride[1]];
				}
			}
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{b->_shape[0],b->_shape[1]};
		ptr->_stride=std::vector<size_t>{b->_shape[1],1};
		return ptr;
	}
	//2D-1D
	else if(a->_shape.size()==2 && b->_shape.size()==1)
	{
		if(a->_shape[1]!=b->_shape[0]) throw std::runtime_error("Shape[1] of 2D tensor must be equal to Shape[0] of 1D tensor for addition!");
		std::vector<dtype> result;
		result.reserve(a->_shape[0] * a->_shape[1]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			for(size_t j=0;j<a->_shape[1];j++)
			{
				result.push_back(a->_data[i * a->_stride[0] + j * a->_stride[1]] - b->_data[j]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			for(size_t i=0;i<a->_shape[0];i++)
			{
				for(size_t j=0;j<a->_shape[1];j++)
				{
					a->_grad[i * a->_stride[0] +j * a->_stride[1]] += grad_out[i * a->_stride[0] + j * a->_stride[1]];
					b->_grad[j]-=grad_out[i * a->_stride[0] + j * a->_stride[1]];
				}
			}
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{a->_shape[0],a->_shape[1]};
		ptr->_stride=std::vector<size_t>{a->_shape[1],1};
		return ptr;
	}
	//2D-2D
	else if(a->_shape.size()==2 && b->_shape.size()==2)
	{
		if(a->_shape!=b->_shape) throw std::runtime_error("Both 2D tensors must be of same shape for addition!");
		std::vector<dtype> result;
		result.reserve(a->_shape[0] * a->_shape[1]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			for(size_t j=0;j<a->_shape[1];j++)
			{
				result.push_back(a->_data[i * a->_stride[0] + j * a->_stride[1]] - b->_data[i * b->_stride[0] + j * b->_stride[1]]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			for(size_t i=0;i<a->_shape[0];i++)
			{
				for(size_t j=0;j<a->_shape[1];j++)
				{
					a->_grad[i * a->_stride[0] +j * a->_stride[1]] += grad_out[i * a->_stride[0] + j * a->_stride[1]];
					b->_grad[i * b->_stride[0] +j * b->_stride[1]] -= grad_out[i * b->_stride[0] + j * b->_stride[1]];
				}
			}
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{b->_shape[0],b->_shape[1]};
		ptr->_stride=std::vector<size_t>{b->_shape[1],1};
		return ptr;
	}
	else
	{
		throw std::runtime_error("Not supported yet!");
	}

}
std::shared_ptr<TensorPtr> TensorPtr::dot(const std::shared_ptr<TensorPtr> &a,const std::shared_ptr<TensorPtr> &b)
{
	if(a->_shape.size()!=1 || b->_shape.size()!=1) throw std::runtime_error("Both tensors must be 1D for dot product!");
	if(a->_shape!=b->_shape) throw std::runtime_error("Inconsistent tensor size(Expected length "+std::to_string(a->_shape[0])+" but got "+std::to_string(b->_shape[0])+"!)");
	dtype result=0;
	for(size_t i=0;i<a->_shape[0];i++)
	{
		result+=a->_data[i] * b->_data[i];
	}
	std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
	std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			for(size_t i=0;i<a->_data.size();i++)
			{
				a->_grad[i] += b->_data[i] * grad_out[0];
				b->_grad[i] += a->_data[i] * grad_out[0];
			}
		};
	return std::make_shared<TensorPtr>(result,parents,grad_fn);
}
std::shared_ptr<TensorPtr> TensorPtr::elementwise_mul(const std::shared_ptr<TensorPtr> &a,const std::shared_ptr<TensorPtr> &b)
{	
	//scalar * scalar
	if(a->_shape.size()==0 && b->_shape.size()==0)
	{
		dtype result=a->_data[0]*b->_data[0];
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			a->_grad[0]+=b->_data[0]*grad_out[0];
			b->_grad[0]+=a->_data[0]*grad_out[0];
		};
		return std::make_shared<TensorPtr>(result,parents,grad_fn);
	}
	
	//scalar * 1D
	else if(a->_shape.size()==0 && b->_shape.size()==1)
	{
		std::vector<dtype> result;
		result.reserve(b->_shape[0]);
		for(size_t i=0;i<b->_shape[0];i++)
		{
			result.push_back(a->_data[0]*b->_data[i]);
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			for(size_t i=0;i<b->_shape[0];i++)
			{
				a->_grad[0] += b->_data[i]*grad_out[i];
				b->_grad[i] += a->_data[0] * grad_out[i];
			}
		};
		return std::make_shared<TensorPtr>(result,parents,grad_fn);
	}
	//1D * scalar
	else if(a->_shape.size()==1 && b->_shape.size()==0)
	{
		std::vector<dtype> result;
		result.reserve(a->_shape[0]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			result.push_back(a->_data[i]*b->_data[0]);
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			for(size_t i=0;i<a->_shape[0];i++)
			{
				a->_grad[i] += b->_data[0] * grad_out[i];
				b->_grad[0] += a->_data[i] * grad_out[i];
			}
		};
		return std::make_shared<TensorPtr>(result,parents,grad_fn);
	}
	//1D*1D
	else if(a->_shape.size()==1 && b->_shape.size()==1)
	{
		if(a->_shape!=b->_shape) throw std::runtime_error("Both 1D tensors must be of same shape for multiplication!");
		std::vector<dtype> result;
		result.reserve(a->_shape[0]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			result.push_back(a->_data[i]*b->_data[i]);
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype>&)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
		    for(size_t i=0;i<a->_shape[0];i++)
		    {
		        a->_grad[i]+=b->_data[i]*grad_out[i];
		        b->_grad[i]+=a->_data[i]*grad_out[i];
		    }
		};
		return std::make_shared<TensorPtr>(result,parents,grad_fn);
	}
	//scalar * 2D
	else if(a->_shape.size()==0 && b->_shape.size()==2)
	{
		std::vector<dtype> result;
		result.reserve(b->_shape[0] * b->_shape[1]);
		for(size_t i=0;i<b->_shape[0];i++)
		{
			for(size_t j=0;j<b->_shape[1];j++)
			{
				result.push_back(a->_data[0]*b->_data[i * b->_stride[0] + j * b->_stride[1]]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype>&)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
		    for(size_t i=0;i<b->_shape[0];i++)
		        for(size_t j=0;j<b->_shape[1];j++)
		        {
		            dtype bij=b->_data[i * b->_stride[0]+j * b->_stride[1]];
		            a->_grad[0] += b->_data[i * b->_stride[0]+j * b->_stride[1]] * grad_out[i * b->_shape[1]+j];
		            b->_grad[i * b->_stride[0] + j * b->_stride[1]] += a->_data[0] * grad_out[i * b->_shape[1] + j];
		        }
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{b->_shape[0],b->_shape[1]};
		ptr->_stride=std::vector<size_t>{b->_shape[1],1};
		return ptr;
	}
	//2D*scalar
	else if(a->_shape.size()==2 && b->_shape.size()==0)
	{
		std::vector<dtype> result;
		result.reserve(a->_shape[0] * a->_shape[1]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			for(size_t j=0;j<a->_shape[1];j++)
			{
				result.push_back(a->_data[i * a->_stride[0] + j * a->_stride[1]] * b->_data[0]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype>&)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
		    for(size_t i=0;i<a->_shape[0];i++)
		        for(size_t j=0;j<a->_shape[1];j++)
		        {
		            a->_grad[i*a->_stride[0]+j*a->_stride[1]] += b->_data[0] * grad_out[i*a->_shape[1]+j];
		            b->_grad[0] += a->_data[i*a->_stride[0]+j*a->_stride[1]] * grad_out[i*a->_shape[1]+j];
		        }
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{a->_shape[0],a->_shape[1]};
		ptr->_stride=std::vector<size_t>{a->_shape[1],1};
		return ptr;
	}
	//1D*2D
	else if(a->_shape.size()==1 && b->_shape.size()==2)
	{
		if(a->_shape[0]!=b->_shape[1]) throw std::runtime_error("Shape[0] of 1D tensor must be equal to Shape[1] of 2D tensor for multiplication!");
		std::vector<dtype> result;
		result.reserve(b->_shape[0] * b->_shape[1]);
		for(size_t i=0;i<b->_shape[0];i++)
		{
			for(size_t j=0;j<b->_shape[1];j++)
			{
				result.push_back(a->_data[j] * b->_data[i * b->_stride[0] + j * b->_stride[1]]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype>&)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
		    for(size_t i=0;i<b->_shape[0];i++)
		        for(size_t j=0;j<b->_shape[1];j++)
		        {
		            a->_grad[j] += b->_data[i*b->_stride[0]+j*b->_stride[1]] * grad_out[i * b->_stride[0] + j *b->_stride[1]];
		            b->_grad[i*b->_stride[0]+j*b->_stride[1]] += a->_data[j] * grad_out[i * b->_shape[1] + j];
		        }
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{b->_shape[0],b->_shape[1]};
		ptr->_stride=std::vector<size_t>{b->_shape[1],1};
		return ptr;
	}
	//2D*1D
	else if(a->_shape.size()==2 && b->_shape.size()==1)
	{
		if(a->_shape[1]!=b->_shape[0]) throw std::runtime_error("Shape[1] of 2D tensor must be equal to Shape[0] of 1D tensor for multiplication!");
		std::vector<dtype> result;
		result.reserve(a->_shape[0] * a->_shape[1]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			for(size_t j=0;j<a->_shape[1];j++)
			{
				result.push_back(a->_data[i * a->_stride[0] + j * a->_stride[1]] * b->_data[j]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype>&)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
		    for(size_t i=0;i<a->_shape[0];i++)
		        for(size_t j=0;j<a->_shape[1];j++)
		        {
		            a->_grad[i*a->_stride[0]+j*a->_stride[1]] += b->_data[j] * grad_out[i*a->_stride[0]+j];
		            b->_grad[j] += a->_data[i*a->_stride[0]+j*a->_stride[1]] * grad_out[i*a->_stride[0]+j];
		        }
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{a->_shape[0],a->_shape[1]};
		ptr->_stride=std::vector<size_t>{a->_shape[1],1};
		return ptr;
	}
	//2D*2D
	else if(a->_shape.size()==2 && b->_shape.size()==2)
	{
		if(a->_shape!=b->_shape) throw std::runtime_error("Both 2D tensors must have same size for multiplication!");
		std::vector<dtype> result;
		result.reserve(a->_shape[0] * a->_shape[1]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			for(size_t j=0;j<a->_shape[1];j++)
			{
				result.push_back(a->_data[i * a->_stride[0] + j * a->_stride[1]] * b->_data[i * b->_stride[0] + j * b->_stride[1]]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype>&)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
		    for(size_t i=0;i<a->_shape[0];i++)
		        for(size_t j=0;j<a->_shape[1];j++)
		        {
		            a->_grad[i*a->_stride[0]+j*a->_stride[1]] += b->_data[i*b->_stride[0]+j*b->_stride[1]] * grad_out[i*a->_shape[1]+j];
		            b->_grad[i*b->_stride[0]+j*b->_stride[1]] += a->_data[i*a->_stride[0]+j*a->_stride[1]] * grad_out[i*a->_shape[1]+j];
		        }
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{b->_shape[0],b->_shape[1]};
		ptr->_stride=std::vector<size_t>{b->_shape[1],1};
		return ptr;
	}
	else
	{
		throw std::runtime_error("Not supported yet!");
	}
}

std::shared_ptr<TensorPtr> TensorPtr::matmul(const std::shared_ptr<TensorPtr> &a,const std::shared_ptr<TensorPtr> &b)
{
	//If any Tensor is 0D
	if(a->_shape.size()==0 || b->_shape.size()==0)
	{
		throw std::runtime_error("Both tensors must be atleast 1D but got "+std::to_string(a->_shape.size())+"D and "+std::to_string(b->_shape.size())+"D tensors!");
	}
	//1D @ 1D
	else if(a->_shape.size()==1 && b->_shape.size()==1)
	{
		return dot(a,b);
	}
	//1D @ 2D
	else if(a->_shape.size()==1 && b->_shape.size()==2)
	{
		if(a->_shape[0]!=b->_shape[0]) throw std::runtime_error("Shape[0] of 1D tensor must be equal to Shape[0] of 2D tensor for matrix multiplication!");
	    dtype ans;
	    std::vector<dtype> result;
	    result.reserve(b->_shape[1]);
	    for(size_t i=0;i<b->_shape[1];i++)
	    {
	        ans=0;
	        for(size_t j=0;j<a->_shape[0];j++)
	            ans+=a->_data[j] * b->_data[j * b->_stride[0] + i * b->_stride[1]];
	        result.push_back(ans);
	    }

		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype>&)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
	        for(size_t i=0;i<a->_shape[0];i++)
	        {
	            dtype ga=0;
	            for(size_t j=0;j<b->_shape[1];j++)
	            {
	                dtype g=grad_out[j];
	                ga+=b->_data[i*b->_stride[0]+j*b->_stride[1]]*g;
	                b->_grad[i*b->_stride[0]+j*b->_stride[1]]+=a->_data[i]*g;
	            }
	            a->_grad[i]+=ga;
	        }
	    };
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{b->_shape[1]};
		ptr->_stride=std::vector<size_t>{1};
		return ptr;
	}
	//2D @ 1D
	else if(a->_shape.size()==2 && b->_shape.size()==1)
	{
		if(a->_shape[1]!=b->_shape[0]) throw std::runtime_error("Shape[1] of 2D tensor must be equal to Shape[0] of 1D tensor for matrix multiplication!");
		dtype ans;
		std::vector<dtype> result;
		result.reserve(a->_shape[0]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			ans=0;
			for(size_t j=0;j<a->_shape[1];j++)
			{
				ans+=a->_data[i * a->_stride[0] + j * a->_stride[1]] * b->_data[j];
			}
			result.push_back(ans);
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype>&)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
	        for(size_t i=0;i<a->_shape[0];i++)
	        {
	            for(size_t j=0;j<a->_shape[1];j++)
	            {
	                a->_grad[i * a->_stride[0] + j * a->_stride[1]] += grad_out[i] * b->_data[j];
	                b->_grad[j] += a->_data[i * a->_stride[0] + j * a->_stride[1]] * grad_out[i];
	            }
	        }
	    };
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{a->_shape[0]};
		ptr->_stride=std::vector<size_t>{1};
		return ptr;
	}
	//2D @ 2D
	else if(a->_shape.size()==2 && b->_shape.size()==2)
	{
		if(a->_shape[1]!=b->_shape[0]) throw std::runtime_error("Shape[1] of 2D tensor must be equal to Shape[0] of 2D tensor for matrix multiplication!");
		dtype ans;
		std::vector<dtype> result;
		result.reserve(a->_shape[0] * b->_shape[1]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			for(size_t j=0;j<b->_shape[1];j++)
			{
				ans=0;
				for(size_t k=0;k<a->_shape[1];k++)
				{
					ans+=a->_data[i * a->_stride[0] + k * a->_stride[1]] * b->_data[k * b->_stride[0] + j * b->_stride[1]];
				}
				result.push_back(ans);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype>&)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
	        for(size_t i=0;i<a->_shape[0];i++)
	            for(size_t j=0;j<b->_shape[1];j++)
	            {
	                for(size_t k=0;k<a->_shape[1];k++)
	                {
	                    a->_grad[i * a->_stride[0] + k * a->_stride[1]] += grad_out[i * b->_shape[1] + j] * b->_data[k * b->_stride[0] + j * b->_stride[1]];
	                    b->_grad[k * b->_stride[0] + j * b->_stride[1]] += a->_data[i * a->_stride[0] + k * a->_stride[1]] * grad_out[i * b->_shape[1] + j];
	                }
	            }
	    };
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{a->_shape[0],b->_shape[1]};
		ptr->_stride=std::vector<size_t>{b->_shape[1],1};
		return ptr;
	}
	else
	{
		throw std::runtime_error("Not supported yet!");
	}
}

std::shared_ptr<TensorPtr> TensorPtr::div(const std::shared_ptr<TensorPtr> &a,const std::shared_ptr<TensorPtr> &b)
{	
	//scalar / scalar
	if(a->_shape.size()==0 && b->_shape.size()==0)
	{
		dtype result=a->_data[0]/b->_data[0];
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			dtype bval=b->_data[0];
			a->_grad[0]+=grad_out[0]/bval;
			b->_grad[0]+=-a->_data[0]/(bval*bval)*grad_out[0];
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		return ptr;
	}
	
	//scalar / 1D
	else if(a->_shape.size()==0 && b->_shape.size()==1)
	{
		std::vector<dtype> result;
		result.reserve(b->_shape[0]);
		for(size_t i=0;i<b->_shape[0];i++)
		{
			result.push_back(a->_data[0]/b->_data[i]);
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			for(size_t i=0;i<b->_shape[0];i++)
			{
				dtype bval=b->_data[i];
				a->_grad[0]+=grad_out[i]/bval;
				b->_grad[i]+=-a->_data[0]/(bval*bval)*grad_out[i];
			}
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		return ptr;
	}
	//1D / scalar
	else if(a->_shape.size()==1 && b->_shape.size()==0)
	{
		std::vector<dtype> result;
		result.reserve(a->_shape[0]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			result.push_back(a->_data[i]/b->_data[0]);
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			dtype bval=b->_data[0];
			for(size_t i=0;i<a->_shape[0];i++)
			{
				a->_grad[i]+=grad_out[i]/bval;
				b->_grad[0]+=-a->_data[i]/(bval*bval)*grad_out[i];
			}
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		return ptr;
	}
	//1D/1D
	else if(a->_shape.size()==1 && b->_shape.size()==1)
	{
		if(a->_shape!=b->_shape) throw std::runtime_error("Both 1D tensors must be of same shape for division!");
		std::vector<dtype> result;
		result.reserve(a->_shape[0]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			result.push_back(a->_data[i]/b->_data[i]);
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			for(size_t i=0;i<a->_shape[0];i++)
			{
				dtype bval=b->_data[i];
				a->_grad[i]+=grad_out[i]/bval;
				b->_grad[i]+=-a->_data[i]/(bval*bval)*grad_out[i];
			}
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		return ptr;
	}
	//scalar / 2D
	else if(a->_shape.size()==0 && b->_shape.size()==2)
	{
		std::vector<dtype> result;
		result.reserve(b->_shape[0] * b->_shape[1]);
		for(size_t i=0;i<b->_shape[0];i++)
		{
			for(size_t j=0;j<b->_shape[1];j++)
			{
				result.push_back(a->_data[0]/b->_data[i * b->_stride[0] + j * b->_stride[1]]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			size_t cols=b->_shape[1];
			for(size_t i=0;i<b->_shape[0];i++)
			{
				for(size_t j=0;j<cols;j++)
				{
					dtype bval=b->_data[i * b->_stride[0] + j * b->_stride[1]];
					dtype g=grad_out[i*cols+j];
					a->_grad[0]+=g/bval;
					b->_grad[i * b->_stride[0] + j * b->_stride[1]]+=-a->_data[0]/(bval*bval)*g;
				}
			}
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{b->_shape[0],b->_shape[1]};
		ptr->_stride=std::vector<size_t>{b->_shape[1],1};
		return ptr;
	}
	//2D / scalar
	else if(a->_shape.size()==2 && b->_shape.size()==0)
	{
		std::vector<dtype> result;
		result.reserve(a->_shape[0] * a->_shape[1]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			for(size_t j=0;j<a->_shape[1];j++)
			{
				result.push_back(a->_data[i * a->_stride[0] + j * a->_stride[1]] / b->_data[0]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			size_t cols=a->_shape[1];
			dtype bval=b->_data[0];
			for(size_t i=0;i<a->_shape[0];i++)
			{
				for(size_t j=0;j<cols;j++)
				{
					dtype g=grad_out[i*cols+j];
					a->_grad[i * a->_stride[0] + j * a->_stride[1]]+=g/bval;
					b->_grad[0]+=-a->_data[i * a->_stride[0] + j * a->_stride[1]]/(bval*bval)*g;
				}
			}
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{a->_shape[0],a->_shape[1]};
		ptr->_stride=std::vector<size_t>{a->_shape[1],1};
		return ptr;
	}
	//1D/2D
	else if(a->_shape.size()==1 && b->_shape.size()==2)
	{
		if(a->_shape[0]!=b->_shape[1]) throw std::runtime_error("Shape[0] of 1D tensor must be equal to Shape[1] of 2D tensor for division!");
		std::vector<dtype> result;
		result.reserve(b->_shape[0] * b->_shape[1]);
		for(size_t i=0;i<b->_shape[0];i++)
		{
			for(size_t j=0;j<b->_shape[1];j++)
			{
				result.push_back(a->_data[j] / b->_data[i * b->_stride[0] + j * b->_stride[1]]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			size_t cols=b->_shape[1];
			for(size_t i=0;i<b->_shape[0];i++)
			{
				for(size_t j=0;j<cols;j++)
				{
					dtype bval=b->_data[i * b->_stride[0] + j * b->_stride[1]];
					dtype g=grad_out[i*cols+j];
					a->_grad[j]+=g/bval;
					b->_grad[i * b->_stride[0] + j * b->_stride[1]]+=-a->_data[j]/(bval*bval)*g;
				}
			}
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{b->_shape[0],b->_shape[1]};
		ptr->_stride=std::vector<size_t>{b->_shape[1],1};
		return ptr;
	}
	//2D/1D
	else if(a->_shape.size()==2 && b->_shape.size()==1)
	{
		if(a->_shape[1]!=b->_shape[0]) throw std::runtime_error("Shape[1] of 2D tensor must be equal to Shape[0] of 1D tensor for division!");
		std::vector<dtype> result;
		result.reserve(a->_shape[0] * a->_shape[1]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			for(size_t j=0;j<a->_shape[1];j++)
			{
				result.push_back(a->_data[i * a->_stride[0] + j * a->_stride[1]] / b->_data[j]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			size_t cols=a->_shape[1];
			for(size_t i=0;i<a->_shape[0];i++)
			{
				for(size_t j=0;j<cols;j++)
				{
					dtype bval=b->_data[j];
					dtype g=grad_out[i*cols+j];
					a->_grad[i * a->_stride[0] + j * a->_stride[1]]+=g/bval;
					b->_grad[j]+=-a->_data[i * a->_stride[0] + j * a->_stride[1]]/(bval*bval)*g;
				}
			}
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{a->_shape[0],a->_shape[1]};
		ptr->_stride=std::vector<size_t>{a->_shape[1],1};
		return ptr;
	}
	//2D/2D
	else if(a->_shape.size()==2 && b->_shape.size()==2)
	{
		if(a->_shape!=b->_shape) throw std::runtime_error("Both 2D tensors must be of same shape for division!");
		std::vector<dtype> result;
		result.reserve(a->_shape[0] * a->_shape[1]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			for(size_t j=0;j<a->_shape[1];j++)
			{
				result.push_back(a->_data[i * a->_stride[0] + j * a->_stride[1]] / b->_data[i * b->_stride[0] + j * b->_stride[1]]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			size_t cols=a->_shape[1];
			for(size_t i=0;i<a->_shape[0];i++)
			{
				for(size_t j=0;j<cols;j++)
				{
					dtype bval=b->_data[i * b->_stride[0] + j * b->_stride[1]];
					dtype g=grad_out[i*cols+j];
					a->_grad[i * a->_stride[0] + j * a->_stride[1]]+=g/bval;
					b->_grad[i * b->_stride[0] + j * b->_stride[1]]+=-a->_data[i * a->_stride[0] + j * a->_stride[1]]/(bval*bval)*g;
				}
			}
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{b->_shape[0],b->_shape[1]};
		ptr->_stride=std::vector<size_t>{b->_shape[1],1};
		return ptr;
	}
	else
	{
		throw std::runtime_error("Not supported yet!");
	}
}

std::shared_ptr<TensorPtr> TensorPtr::negation(const std::shared_ptr<TensorPtr> &a)
{
	//scalar
	if(a->_shape.size()==0)
	{
		dtype result=-a->_data[0];
		std::vector<std::shared_ptr<TensorPtr>> parents={a};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a](const std::vector<dtype>& grad_out){
			a->_grad[0]-=grad_out[0];
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		return ptr;
	}
	//1D
	else if(a->_shape.size()==1)
	{
		std::vector<dtype> result;
		result.reserve(a->_shape[0]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			result.push_back(-a->_data[i]);
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a](const std::vector<dtype>& grad_out){
			std::transform(a->_grad.begin(),a->_grad.end(),grad_out.begin(),a->_grad.begin(),std::minus<dtype>());
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		return ptr;
	}
	//2D
	else if(a->_shape.size()==2)
	{
		std::vector<dtype> result;
		result.reserve(a->_shape[0] * a->_shape[1]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			for(size_t j=0;j<a->_shape[1];j++)
			{
				result.push_back(-a->_data[i * a->_stride[0] + j * a->_stride[1]]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a](const std::vector<dtype>& grad_out){
			size_t cols=a->_shape[1];
			for(size_t i=0;i<a->_shape[0];i++)
			{
				for(size_t j=0;j<cols;j++)
				{
					a->_grad[i * a->_stride[0] + j * a->_stride[1]]-=grad_out[i*cols+j];
				}
			}
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{a->_shape[0],a->_shape[1]};
		ptr->_stride=std::vector<size_t>{a->_shape[1],1};
		return ptr;
	}
	else
	{
		throw std::runtime_error("Not supported yet!");
	}
}
std::shared_ptr<TensorPtr> TensorPtr::pow(const std::shared_ptr<TensorPtr> &a,dtype b)
{
	std::vector<dtype> result;
	result.reserve(a->_data.size());
	for(size_t i=0;i<a->_data.size();i++)
	{
		result.push_back(std::pow(a->_data[i],b));
	}
	std::vector<std::shared_ptr<TensorPtr>> parents={a};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			for(size_t i=0;i<a->_data.size();i++)
			{
				a->_grad[i]+=b*std::pow(a->_data[i],b-1);
			}
		};
	std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
	ptr->_shape=a->_shape;
	ptr->_stride=a->_stride;
	return ptr;
}
std::shared_ptr<TensorPtr> TensorPtr::maximum(const std::shared_ptr<TensorPtr> &a,const std::shared_ptr<TensorPtr> &b)
{
	if(a->_shape.size()==0)
	{
		std::vector<dtype> result;
		result.reserve(b->_data.size());
		for(size_t i=0;i<b->_data.size();i++)
		{
			result.push_back(a->_data[0]>=b->_data[i]?a->_data[0]:b->_data[i]);
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			for(size_t i=0;i<b->_data.size();i++)
			{
				if(a->_data[0]>=b->_data[i]) a->_grad[0]+=grad_out[i];
				else b->_grad[i]+=grad_out[i];
			}
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=b->_shape;
		ptr->_stride=b->_stride;
		return ptr;
	}
	else if(b->_shape.size()==0)
	{
		std::vector<dtype> result;
		result.reserve(a->_data.size());
		for(size_t i=0;i<a->_data.size();i++)
		{
			result.push_back(a->_data[i]>=b->_data[0]?a->_data[i]:b->_data[0]);
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a,b};
		std::function<void(const std::vector<dtype> &)> grad_fn=[a,b](const std::vector<dtype>& grad_out){
			for(size_t i=0;i<a->_data.size();i++)
			{
				if(a->_data[i]>=b->_data[0]) a->_grad[i]+=grad_out[i];
				else b->_grad[0]+=grad_out[i];
			}
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=a->_shape;
		ptr->_stride=a->_stride;
		return ptr;
	}
	else throw std::runtime_error("Tensor shape not supported yet!");
}

std::shared_ptr<TensorPtr> TensorPtr::mean(const std::shared_ptr<TensorPtr> &a)
{
	dtype result=0;
	for(size_t i=0;i<a->_data.size();i++)
	{
		result+=a->_data[i]/a->_data.size();
		
	}
	std::vector<std::shared_ptr<TensorPtr>> parents={a};
	std::function<void(const std::vector<dtype>&)> grad_fn=[a](const std::vector<dtype>& grad_out){
			for(size_t i=0;i<a->_data.size();i++)
			{
				a->_grad[i]+=grad_out[i]/a->_data.size();
			}	
		};
	std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
	return ptr;
}

std::shared_ptr<TensorPtr> TensorPtr::transpose(const std::shared_ptr<TensorPtr> &a)
{
	if(a->_shape.size()==0 || a->_shape.size()==1) return a;
	else if(a->_shape.size()==2)
	{
		std::vector<dtype> result;
		result.reserve(a->_data.size());
		for(size_t i=0;i<a->_shape[1];i++)
		{
			for(size_t j=0;j<a->_shape[0];j++)
			{
				result.push_back(a->_data[j * a->_stride[0] + i * a->_stride[1]]);
			}
		}
		std::vector<std::shared_ptr<TensorPtr>> parents={a};
		std::function<void(const std::vector<dtype>&)> grad_fn=[a](const std::vector<dtype>& grad_out){
			for(size_t i=0;i<a->_shape[1];i++)
				for(size_t j=0;j<a->_shape[0];j++)
					a->_grad[j * a->_stride[0] + i * a->_stride[1]] += grad_out[i * a->_shape[0] + j];
		};
		std::shared_ptr<TensorPtr> ptr=std::make_shared<TensorPtr>(result,parents,grad_fn);
		ptr->_shape=std::vector<size_t>{a->_shape[1],a->_shape[0]};
		ptr->_stride=std::vector<size_t>{a->_shape[0],1};
		return ptr;
	}
	else
	{
		throw std::runtime_error("Not supported yet!");
	}
} 

void TensorPtr::zero_grad()
{
	_grad.assign(_data.size(),0);
}
void TensorPtr::reset_graph_visit()
{
	if(!_visited)
	{
		return;
	}
	_visited=false;
	for(size_t i=0;i<_parents.size();i++)
	{
		_parents[i]->reset_graph_visit();
	}
}
void TensorPtr::backward()
{
    if(_shape.size()!=0) throw std::runtime_error("backward() is callable only for a scalar!");
    reset_graph_visit();
    std::vector<TensorPtr*> topo;
    std::function<void(TensorPtr*)> build=[&](TensorPtr* ptr){
        if(ptr->_visited) return;
        ptr->_visited=true;
        for(auto& p : ptr->_parents) build(p.get());
        topo.push_back(ptr);
    };
    build(this);
    _grad={1};
    for(auto it=topo.rbegin(); it!=topo.rend(); ++it)
    {
    	if((*it)->_gradfn) (*it)->_gradfn((*it)->_grad);
	}
}




//Tensor Definition
Tensor::Tensor() {}
Tensor::Tensor(dtype scalar)
{
	ptr=std::shared_ptr<TensorPtr>(new TensorPtr(scalar));
}
Tensor::Tensor(const std::vector<dtype> &vec)
{
	ptr=std::shared_ptr<TensorPtr>(new TensorPtr(vec));
}
Tensor::Tensor(const std::vector<std::vector<dtype>> &vec)
{
	ptr=std::shared_ptr<TensorPtr>(new TensorPtr(vec));
}
Tensor Tensor::randm(const std::initializer_list<size_t> &shape)
{
	if(shape.size()==0) throw std::runtime_error("Shape can't be {}!");
	else if(shape.size()==1)
	{
		size_t size=*shape.begin();
		std::vector<dtype> vec;
		vec.reserve(size);
		for(size_t i=0;i<size;i++)
		{
			vec.push_back(randVal());
		}
		return Tensor(vec);
	}
	else if(shape.size()==2)
	{
		size_t size1=*shape.begin();
		size_t size2=*(shape.begin()+1);
		std::vector<std::vector<dtype>> vec1;
		std::vector<dtype> vec2(size2,0);
		vec1.reserve(size1);
		for(size_t i=0;i<size1;i++)
		{
			for(size_t j=0;j<size2;j++)
			{
				vec2[j]=randVal();
			}
			vec1.push_back(vec2);
		}
		return Tensor(vec1);
	}
	else throw std::runtime_error("Not supported yet!");
}
Tensor Tensor::randm(const std::vector<size_t> &shape)
{
	if(shape.size()==0) throw std::runtime_error("Shape can't be {}!");
	else if(shape.size()==1)
	{
		size_t size=shape[0];
		std::vector<dtype> vec;
		vec.reserve(size);
		for(size_t i=0;i<size;i++)
		{
			vec.push_back(randVal());
		}
		return Tensor(vec);
	}
	else if(shape.size()==2)
	{
		size_t size1=shape[0];
		size_t size2=shape[1];
		std::vector<std::vector<dtype>> vec1;
		std::vector<dtype> vec2(size2,0);
		vec1.reserve(size1);
		for(size_t i=0;i<size1;i++)
		{
			for(size_t j=0;j<size2;j++)
			{
				vec2[j]=randVal();
			}
			vec1.push_back(vec2);
		}
		return Tensor(vec1);
	}
	else throw std::runtime_error("Not supported yet!");
}
Tensor Tensor::zeros(const std::initializer_list<size_t> &shape)
{
	if(shape.size()==0) throw std::runtime_error("Shape can't be {}!");
	else if(shape.size()==1)
	{
		size_t size=*shape.begin();
		std::vector<dtype> vec(size,0);
		return Tensor(vec);
	}
	else if(shape.size()==2)
	{
		size_t size1=*shape.begin();
		size_t size2=*(shape.begin()+1);
		std::vector<std::vector<dtype>> vec1;
		std::vector<dtype> vec2(size2,0);
		vec1.reserve(size1);
		for(size_t i=0;i<size1;i++)
		{
			vec1.push_back(vec2);
		}
		return Tensor(vec1);
	}
	else throw std::runtime_error("Not supported yet!");
}
Tensor Tensor::zeros(const std::vector<size_t> &shape)
{
	if(shape.size()==0) throw std::runtime_error("Shape can't be {}!");
	else if(shape.size()==1)
	{
		size_t size=shape[0];
		std::vector<dtype> vec(size,0);
		return Tensor(vec);
	}
	else if(shape.size()==2)
	{
		size_t size1=shape[0];
		size_t size2=shape[1];
		std::vector<std::vector<dtype>> vec1;
		std::vector<dtype> vec2(size2,0);
		vec1.reserve(size1);
		for(size_t i=0;i<size1;i++)
		{
			vec1.push_back(vec2);
		}
		return Tensor(vec1);
	}
	else throw std::runtime_error("Not supported yet!");
}
std::vector<dtype> & Tensor::data() const
{
	return ptr->_data;
}
std::shared_ptr<TensorPtr> Tensor::get() const
{
	return ptr;
}
const std::vector<size_t> & Tensor::shape() const
{
	return ptr->_shape;
}
const std::vector<size_t> & Tensor::stride() const
{
	return ptr->_stride;
}
size_t Tensor::size()
{
	if(ptr->_shape.size()==0) return 0;
	else return ptr->_shape[0];
}
dtype & Tensor::item() const
{
	if(ptr->_shape.size()!=0) throw std::invalid_argument("item() can be called only on 0D tensors!");
	return ptr->_data[0]; 
}
size_t Tensor::numel() const
{
	return ptr->_data.size();
}

dtype & Tensor::operator()(size_t i) const
{
	if(ptr->_shape.size()==0) throw std::invalid_argument("Use item() for 0D tensors!");
	else if(ptr->_shape.size()>=2) throw std::invalid_argument("Invalid index given!");
	if(i>=ptr->_data.size()) throw std::out_of_range("Index out of range!");
	return ptr->_data[i];
}
dtype & Tensor::operator()(size_t i,size_t j) const
{
	if(ptr->_shape.size()==0) throw std::invalid_argument("Use item() for 0D tensors!");
	else if(ptr->_shape.size()!=2) throw std::invalid_argument("Invalid index given!");
	if(i>=ptr->_shape[0] || j>=ptr->_shape[1]) throw std::out_of_range("Index out of range!");
	return ptr->_data[i*ptr->_stride[0]+j*ptr->_stride[1]];
}
Tensor Tensor::operator[](size_t i)
{
	if(ptr->_shape.size()==0) throw std::invalid_argument("Use item() for 0D tensors!");
	else if(ptr->_shape.size()==1) throw std::invalid_argument("Use () operator for 1D tensors!");
	else if(ptr->_shape.size()==2)
	{
		std::vector<dtype> result;
		result.reserve(ptr->_shape[1]);
		for(size_t j=0;j<ptr->_shape[1];j++)
		{
			result.push_back(ptr->_data[i*ptr->_shape[1]+j]);
		}
		return Tensor(result);
	}
	else throw std::runtime_error("Not supported yet!");
}
Tensor Tensor::operator+(const Tensor &other)
{
	Tensor result;
	result.ptr=TensorPtr::add(ptr,other.ptr);
	return result;
}
Tensor Tensor::operator-(const Tensor &other)
{
	Tensor result;
	result.ptr=TensorPtr::sub(ptr,other.ptr);
	return result;
}
Tensor Tensor::dot(const Tensor &other)
{
	Tensor result;
	result.ptr=TensorPtr::dot(ptr,other.ptr);
	return result;
}
Tensor Tensor::operator*(const Tensor &other)
{
	Tensor result;
	result.ptr=TensorPtr::elementwise_mul(ptr,other.ptr);
	return result;
}
Tensor Tensor::matmul(const Tensor &other) const
{
	Tensor result;
	result.ptr=TensorPtr::matmul(ptr,other.ptr);
	return result;
}
Tensor Tensor::operator/(const Tensor &other)
{
	Tensor result;
	result.ptr=TensorPtr::div(ptr,other.ptr);
	return result;
}
Tensor Tensor::operator-() const
{
	Tensor result;
	result.ptr=TensorPtr::negation(ptr);
	return result;
}
Tensor Tensor::maximum(const Tensor &x,const Tensor &y)
{
	Tensor result;
	result.ptr=TensorPtr::maximum(x.ptr,y.ptr);
	return result;
}
Tensor Tensor::pow(const Tensor &a,dtype b)
{
	Tensor result;
	result.ptr=TensorPtr::pow(a.ptr,b);
	return result;
}
Tensor Tensor::mean(const Tensor &x)
{
	Tensor result;
	result.ptr=TensorPtr::mean(x.ptr);
	return result;
}
Tensor Tensor::T()
{
	Tensor result;
	result.ptr=TensorPtr::transpose(ptr);
	return result;
}
void Tensor::zero_grad() const
{
	ptr->zero_grad();
}
const std::vector<dtype>& Tensor::grad() const
{
	return ptr->_grad;
}
void Tensor::backward()
{
	ptr->backward();
}
std::ostream & operator<<(std::ostream &os,const Tensor &tensor)
{
	if(tensor.shape().size()==0)
	{
		os<<tensor.item();
	}
	else if(tensor.shape().size()==1)
	{
		os<<"{";
		for(size_t i=0;i<tensor.shape()[0];i++)
		{
			os<<tensor.data()[i];
			if(i!=tensor.shape()[0]-1) os<<", ";
		}
		os<<"}";
	}
	else if(tensor.shape().size()==2)
	{
		os<<"{";
		for(size_t i=0;i<tensor.shape()[0];i++)
		{
			os<<"{";
			for(size_t j=0;j<tensor.shape()[1];j++)
			{
				os<<tensor.data()[i*tensor.stride()[0]+j*tensor.stride()[1]];
				if(j!=tensor.shape()[1]-1) os<<",";
			}
			os<<"}";
			if(i!=tensor.shape()[0]-1) os<<",\n ";
		}
		os<<"}";
	}
	return os;
}
