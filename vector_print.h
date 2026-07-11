#pragma once
#include <ostream>
#include <vector>

std::ostream & operator<<(std::ostream &os,const std::vector<size_t> &vec)
{
	os<<"{";
	for(int i=0;i<vec.size();i++)
	{
		os<<vec[i];
		if(i!=vec.size()-1) os<<", ";
	}
	os<<"}";
	return os;
}