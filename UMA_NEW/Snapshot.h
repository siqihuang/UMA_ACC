#ifndef _SNAPSHOT_
#define _SNAPSHOT_

#include <vector>
#include <iostream>
#include <map>
using namespace std;

class Snapshot{
public:
	Snapshot();
	Snapshot(double threshold);
	virtual ~Snapshot();

	void initData(int size,double threshold,vector<vector<int> > context_key,vector<int> context_value);
	void update_state_GPU(bool mode);
	void propagate_GPU();
	void halucinate_GPU(vector<int> actions_list);
	vector<bool> initMask(vector<int> actions_list);
	void setSignal(vector<bool> observe);
	vector<bool> getLoad();
	vector<vector<bool>> getDir();

	vector<vector<double>> update_weights_GPU(vector<vector<double>> weights);

	//enum Type{DECIDE,EXECUTE};

protected:
	int size;
	double threshold;
	std::map<pair<int,int>,int> context;
};

#endif