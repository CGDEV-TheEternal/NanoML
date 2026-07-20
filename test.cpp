#include "model.h"

class MLP:public Model
{
	public:
	Layer l1,relu,l2;
	MLP()
	{
		l1=Layer::Linear(2,4);
		relu=Layer::ReLU();
		l2=Layer::Linear(4,1);
		Model::add_layers({l1,relu,l2});
	}
	Tensor forward(Tensor x) override
	{
		x=l1(x);
		x=relu(x);
		x=l2(x);
		return x;
	}
};
int main()
{
	MLP model=MLP();
	Optimizer optimizer=Optimizer::SGD(model.parameters(),0.01);
	Loss loss_fn=Loss::MSE();
	Tensor x=Tensor(std::vector<std::vector<dtype>>({{0.0,0.0},{0.0,1.0},{1.0,0.0},{1.0,1.0}}));
	Tensor y=Tensor(std::vector<std::vector<dtype>>({{0.0},{1.0},{1.0},{2.0}}));
	for(size_t epoch=0;epoch<1000;epoch++)
	{
		dtype total_loss=0.0;
		Tensor ansp(0),anst(0);
		for(size_t i=0;i<x.size();i++)
		{
			auto pred=model(x[i]);
			auto loss=loss_fn(pred,y[i]);
			ansp=pred;
			anst=y[i];
			optimizer.zero_grad();
			loss.backward();
			optimizer.step();
			
			total_loss+=loss.item();
		}
		if(epoch%100==0) std::cout<<"Epoch "<<(epoch+1)<<": loss = "<<total_loss<<" ans pred:"<<ansp<<", ans true:"<<anst<<"\n";
	}
}