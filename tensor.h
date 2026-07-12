#pragma once
#include <vector>
#include <ostream>
#include <memory>
using dtype=float;
class TensorPtr
{
	private:
		std::vector<dtype> _data;
		std::vector<size_t> _shape;
		std::vector<size_t> _stride;
		TensorPtr(dtype scalar);
		TensorPtr(const std::vector<dtype> &vec);
		TensorPtr(const std::vector<std::vector<dtype>> &vec);
		static std::shared_ptr<TensorPtr> add(const std::shared_ptr<TensorPtr> &a,const std::shared_ptr<TensorPtr> &b);
		static std::shared_ptr<TensorPtr> dot(const std::shared_ptr<TensorPtr> &a,const std::shared_ptr<TensorPtr> &b);
		static std::shared_ptr<TensorPtr> elementwise_mul(const std::shared_ptr<TensorPtr> &a,const std::shared_ptr<TensorPtr> &b);
		static std::shared_ptr<TensorPtr> matmul(const std::shared_ptr<TensorPtr> &a,const std::shared_ptr<TensorPtr> &b);
		static std::shared_ptr<TensorPtr> transpose(const std::shared_ptr<TensorPtr> &a);
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
		
		const std::vector<size_t> &shape() const;
		const std::vector<size_t> &stride() const;
		dtype &item() const;
		size_t numel() const;
		
		dtype & operator()(size_t i) const;
		dtype & operator()(size_t i,size_t j) const;
		Tensor operator+(const Tensor &other);
		Tensor dot(const Tensor &other);
		Tensor operator*(const Tensor &other);
		Tensor matmul(const Tensor &other);
		Tensor T();
		friend std::ostream & operator<<(std::ostream &os,const Tensor &tensor);
};