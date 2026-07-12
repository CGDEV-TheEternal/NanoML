#include "tensor.h"
#include "vector_print.h"
#include <iostream>

//TensorPtr Definition
TensorPtr::TensorPtr(dtype scalar): _data{scalar}, _shape{}, _stride{} {}

TensorPtr::TensorPtr(const std::vector<dtype> &vec): _data(vec), _shape{vec.size()}, _stride{1} {}

TensorPtr::TensorPtr(const std::vector<std::vector<dtype>> &vec): _shape{vec.size(),vec[0].size()}, _stride{vec[0].size(),1}
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
}

std::shared_ptr<TensorPtr> TensorPtr::add(const std::shared_ptr<TensorPtr> &a,const std::shared_ptr<TensorPtr> &b)
{	
	//scalar + scalar
	if(a->_shape.size()==0 && b->_shape.size()==0)
	{
		dtype result=a->_data[0]+b->_data[0];
		std::shared_ptr<TensorPtr> ptr=std::shared_ptr<TensorPtr>(new TensorPtr(result));
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
		std::shared_ptr<TensorPtr> ptr=std::shared_ptr<TensorPtr>(new TensorPtr(result));
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
		std::shared_ptr<TensorPtr> ptr=std::shared_ptr<TensorPtr>(new TensorPtr(result));
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
		std::shared_ptr<TensorPtr> ptr=std::shared_ptr<TensorPtr>(new TensorPtr(result));
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
		std::shared_ptr<TensorPtr> ptr=std::shared_ptr<TensorPtr>(new TensorPtr(result));
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
		std::shared_ptr<TensorPtr> ptr=std::shared_ptr<TensorPtr>(new TensorPtr(result));
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
		std::shared_ptr<TensorPtr> ptr=std::shared_ptr<TensorPtr>(new TensorPtr(result));
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
		std::shared_ptr<TensorPtr> ptr=std::shared_ptr<TensorPtr>(new TensorPtr(result));
		ptr->_shape=std::vector<size_t>{a->_shape[0],a->_shape[1]};
		ptr->_stride=std::vector<size_t>{a->_shape[1],1};
		return ptr;
	}
	//2D+2D
	else if(a->_shape.size()==2 && b->_shape.size()==2)
	{
		std::vector<dtype> result;
		result.reserve(a->_shape[0] * a->_shape[1]);
		for(size_t i=0;i<a->_shape[0];i++)
		{
			for(size_t j=0;j<a->_shape[1];j++)
			{
				result.push_back(a->_data[i * a->_stride[0] + j * a->_stride[1]] + b->_data[i * b->_stride[0] + j * b->_stride[1]]);
			}
		}
		std::shared_ptr<TensorPtr> ptr=std::shared_ptr<TensorPtr>(new TensorPtr(result));
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
	return std::shared_ptr<TensorPtr>(new TensorPtr(result));
}
std::shared_ptr<TensorPtr> TensorPtr::elementwise_mul(const std::shared_ptr<TensorPtr> &a,const std::shared_ptr<TensorPtr> &b)
{	
	//scalar * scalar
	if(a->_shape.size()==0 && b->_shape.size()==0)
	{
		dtype result=a->_data[0]*b->_data[0];
		return std::shared_ptr<TensorPtr>(new TensorPtr(result));
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
		return std::shared_ptr<TensorPtr>(new TensorPtr(result));
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
		return std::shared_ptr<TensorPtr>(new TensorPtr(result));
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
		return std::shared_ptr<TensorPtr>(new TensorPtr(result));
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
		std::shared_ptr<TensorPtr> ptr=std::shared_ptr<TensorPtr>(new TensorPtr(result));
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
		std::shared_ptr<TensorPtr> ptr=std::shared_ptr<TensorPtr>(new TensorPtr(result));
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
		std::shared_ptr<TensorPtr> ptr=std::shared_ptr<TensorPtr>(new TensorPtr(result));
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
		std::shared_ptr<TensorPtr> ptr=std::shared_ptr<TensorPtr>(new TensorPtr(result));
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
		std::shared_ptr<TensorPtr> ptr=std::shared_ptr<TensorPtr>(new TensorPtr(result));
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
			{
				ans+=a->_data[j] * b->_data[i * b->_stride[0] + j * b->_stride[1]];
			}
			result.push_back(ans);
		}
		std::shared_ptr<TensorPtr> ptr=std::shared_ptr<TensorPtr>(new TensorPtr(result));
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
		std::shared_ptr<TensorPtr> ptr=std::shared_ptr<TensorPtr>(new TensorPtr(result));
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
		std::shared_ptr<TensorPtr> ptr=std::shared_ptr<TensorPtr>(new TensorPtr(result));
		ptr->_shape=std::vector<size_t>{a->_shape[0],b->_shape[1]};
		ptr->_stride=std::vector<size_t>{b->_shape[1],1};
		return ptr;
	}
	else
	{
		throw std::runtime_error("Not supported yet!");
	}
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
		std::shared_ptr<TensorPtr> ptr=std::shared_ptr<TensorPtr>(new TensorPtr(result));
		ptr->_shape=std::vector<size_t>{a->_shape[1],a->_shape[0]};
		ptr->_stride=std::vector<size_t>{a->_shape[0],1};
		return ptr;
	}
	else
	{
		throw std::runtime_error("Not supported yet!");
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
std::vector<dtype> & Tensor::data() const
{
	return ptr->_data;
}
const std::vector<size_t> & Tensor::shape() const
{
	return ptr->_shape;
}
const std::vector<size_t> & Tensor::stride() const
{
	return ptr->_stride;
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
Tensor Tensor::operator+(const Tensor &other)
{
	Tensor result;
	result.ptr=TensorPtr::add(ptr,other.ptr);
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
Tensor Tensor::matmul(const Tensor &other)
{
	Tensor result;
	result.ptr=TensorPtr::matmul(ptr,other.ptr);
	return result;
}
Tensor Tensor::T()
{
	Tensor result;
	result.ptr=TensorPtr::transpose(ptr);
	return result;
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


int main(){
	std::vector<std::vector<float>> vec1={{1,2,3},{4,5,6}};
	std::vector<std::vector<float>> vec2={{7,8},{9,10},{11,12}};
	Tensor a(vec1);
	Tensor b(vec2);

}