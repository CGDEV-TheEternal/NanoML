#pragma once
#include <vector>
#include <ostream>
#include <memory>
#include <functional>
#include "vector_print.h"
using dtype=float;
class TensorPtr
{
	private:
		std::vector<dtype> _data;
		std::vector<size_t> _shape;
		std::vector<size_t> _stride;
		std::vector<std::shared_ptr<TensorPtr>> _parents={};
		std::vector<dtype> _grad={};
		std::function<void(const std::vector<dtype>&)> _gradfn=nullptr;
		bool _visited=false; //the visited is set to true if the tensor has already been visited
		
		TensorPtr(dtype scalar,const std::vector<std::shared_ptr<TensorPtr>> & parents={},std::function<void(const std::vector<dtype> &)> gradfn=nullptr);
		TensorPtr(const std::vector<dtype> &vec,const std::vector<std::shared_ptr<TensorPtr>> & parents={},std::function<void(const std::vector<dtype> &)> gradfn=nullptr);
		TensorPtr(const std::vector<std::vector<dtype>> &vec,const std::vector<std::shared_ptr<TensorPtr>> & parents={},std::function<void(const std::vector<dtype> &)> gradfn=nullptr);
		
		static std::shared_ptr<TensorPtr> add(const std::shared_ptr<TensorPtr> &a,const std::shared_ptr<TensorPtr> &b);
		static std::shared_ptr<TensorPtr> dot(const std::shared_ptr<TensorPtr> &a,const std::shared_ptr<TensorPtr> &b);
		static std::shared_ptr<TensorPtr> elementwise_mul(const std::shared_ptr<TensorPtr> &a,const std::shared_ptr<TensorPtr> &b);
		static std::shared_ptr<TensorPtr> matmul(const std::shared_ptr<TensorPtr> &a,const std::shared_ptr<TensorPtr> &b);
		static std::shared_ptr<TensorPtr> transpose(const std::shared_ptr<TensorPtr> &a);
		
		void zero_grad(); //zeros out the grads of tensors.
		void reset_graph_visit(); //resets the visited member of a tensor		
		void backward(); //performs backward pass to calculate the grad of all tensors
		friend class Tensor;
};
class Tensor
{
	private:
		Tensor();
		std::shared_ptr<TensorPtr> ptr;
		std::vector<dtype> &data() const;
	public:
		Tensor(dtype scalar);
		Tensor(const std::vector<dtype> &vec);
		Tensor(const std::vector<std::vector<dtype>> &vec);
		
		const std::vector<size_t> &shape() const;  //getter method for the shape of tensor
		const std::vector<size_t> &stride() const; //getter method for the stride of tensor
		dtype &item() const;  //accessing element in a scalar tensor
		size_t numel() const;  //no of elements in tensor
		
		dtype & operator()(size_t i) const;  //accessing  element in a 1D tensor
		dtype & operator()(size_t i,size_t j) const;  //accessing element in a 2D tensor
		Tensor operator+(const Tensor &other);  //Adds 2 tensors
		Tensor dot(const Tensor &other);  //Dot product 2 tensors
		Tensor operator*(const Tensor &other);  // Elementwise multiplication of 2 tensors
		Tensor matmul(const Tensor &other);  //Matrix multiplication of 2 tensors
		Tensor T();   //Transpose of a tensor
		void zero_grad();
		const std::vector<dtype>& grad() const;   //getter method for grad of a tensor
		void backward();  //calls backward() on  a scalar
		friend std::ostream & operator<<(std::ostream &os,const Tensor &tensor);
};