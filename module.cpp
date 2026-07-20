#include "module.h"

//Base Class Module
Tensor Module::forward(Tensor x)
{
	throw std::runtime_error("forward() not implemented!");
}
Tensor Module::operator()(Tensor x)
{
	return forward(x);
}

//Class Layer
Layer Layer::Linear(size_t in_features,size_t out_features)
{
	Layer layer;
	layer.layer_type="Linear";
	layer.weights=Tensor::randm({in_features,out_features});
	layer.biases=Tensor::zeros({out_features});
	return layer;
}

Layer Layer::Flatten()
{
	Layer layer;
	layer.layer_type="Flatten";
	return layer;
}

Layer Layer::ReLU()
{
	Layer layer;
	layer.layer_type="ReLU";
	return layer;
}
Tensor Layer::forward(Tensor x)
{
	if(layer_type=="Linear")
	{
		Tensor result=x.matmul(weights)+biases;
		return result;
	}
	else if(layer_type=="Flatten")
	{
		Tensor result=Tensor(x.data());
		return result;
	}
	else if(layer_type=="ReLU")
	{
		Tensor result=Tensor::maximum(Tensor(0),x);
		return result;
	}
	else throw std::runtime_error("No such layer supported!");
}


//Class Loss
Tensor Loss::forward(Tensor actual,Tensor pred)
{
	if(loss_type=="MSE")
	{
		Tensor loss=Tensor::mean(Tensor::pow(actual-pred,2));
		return loss;
	}
	else throw std::runtime_error("No such loss function support yet!");
}
Tensor Loss::operator()(Tensor actual,Tensor pred)
{
	return forward(actual,pred);
}
Loss Loss::MSE()
{
	Loss loss;
	loss.loss_type="MSE";
	return loss;
}


//Class Optimizer
void Optimizer::zero_grad() const
{
	for(const Tensor&p:_params)
	{
		p.zero_grad();
	}
}
void Optimizer::step()
{
	if(optimizer_type=="SGD")
	{
		for(const Tensor&p:_params)
		{
			size_t size=(p.ptr)->_data.size();
			for(size_t i=0;i<size;i++)
			{
				(p.ptr)->_data[i]=(p.ptr)->_data[i] - _lr * (p.ptr)->_grad[i];
			}
		}	
	}
	
}

Optimizer Optimizer::SGD(const std::vector<Tensor> &params,dtype lr)
{
	Optimizer optim;
	optim.optimizer_type="SGD";
	optim._params=params;
	optim._lr=lr;
	return optim;
}