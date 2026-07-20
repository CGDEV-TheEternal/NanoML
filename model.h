#pragma once
#include "module.h"

class Model:public Module
{
	private:
		std::vector<Layer> layers;
	public:
		void add_layers(const std::initializer_list<Layer> &l);
		std::vector<Tensor> parameters();
};