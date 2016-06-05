#ifndef _SNAPSHOT_
#define _SNAPSHOT_

#include <vector>
#include <iostream>
#include <map>
#include <time.h>
using namespace std;

class Snapshot{
public:
	Snapshot();
	Snapshot(double threshold);
	virtual ~Snapshot();

	void initData(string name,int size,double threshold,vector<vector<int> > context_key,vector<int> context_value,vector<string> sensors_names,vector<string> evals_names,vector<vector<int> > generalized_actions);
	void freeData();
	void update_state_GPU(bool mode);
	void propagate_GPU();
	void halucinate_GPU(vector<int> actions_list);
	vector<bool> initMask(vector<int> actions_list);
	void setSignal(vector<bool> observe);
	vector<bool> getCurrent();
	vector<bool> getLoad();
	vector<vector<bool> > getDir();
	vector<bool> halucinate(vector<int> action_list);

protected:
	int size;
	double threshold;
	std::map<pair<int,int>,int> context;
	vector<string> sensors_names,evals_names;
	vector<vector<int> > generalized_actions;
	string name;
	std::map<string,int> name_to_num;
};

#endif