#include "model.h"

void Model::add_layers(const std::initializer_list<Layer> &l)
{
	layers=std::vector<Layer>(l);
}
std::vector<Tensor> Model::parameters()
{
	std::vector<Tensor> _parameters;
	for(const Layer &layer:layers)
	{
		if(layer.layer_type=="Linear")
		{
			_parameters.push_back(layer.weights);
			_parameters.push_back(layer.biases);
		}
	}
	return _parameters;
}