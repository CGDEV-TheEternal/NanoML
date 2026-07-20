#pragma once
#include <iostream>
#include "tensor.h"
class Module
{
	public:
		virtual Tensor forward(Tensor x);
		Tensor operator()(Tensor x);		
};

class Layer: public Module
{
	private:
		Tensor weights,biases;
		std::string layer_type;
		
	public:
		static Layer Linear(size_t in_features,size_t out_features);
		static Layer Flatten();
		static Layer ReLU();
		Tensor forward(Tensor x) override;
		
		friend class Model;
};

class Loss
{
	private:
		Loss(){}
		std::string loss_type;
	public:
		static Loss MSE();  //Mean Squared Error
		Tensor forward(Tensor x,Tensor y);
		Tensor operator()(Tensor x,Tensor y);
};

class Optimizer
{
	private:
		std::string optimizer_type;
		std::vector<Tensor> _params;
		dtype _lr;
		Optimizer(){}
	public:
		void zero_grad() const;
		void step();
		static Optimizer SGD(const std::vector<Tensor> &params,dtype lr);
};