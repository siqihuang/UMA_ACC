#include "Agent.h"

Agent::Agent():Snapshot(){}

Agent::Agent(double threshold):Snapshot(threshold){}

vector<vector<bool>> Agent::halucinate_all(vector<vector<int>> generalized_actions){
	vector<vector<bool>> result;
	for(int i=0;i<generalized_actions.size();++i){
		halucinate_GPU(generalized_actions[i]);
		result.push_back(this->getLoad());
	}
	return result;
}

vector<bool> Agent::halucinate(vector<int> action_list){
	halucinate_GPU(action_list);
	return this->getLoad();
}

Agent::~Agent(){}