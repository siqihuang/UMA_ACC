#include "Snapshot.h"

Snapshot::Snapshot(){}

Snapshot::Snapshot(double threshold){
	this->threshold=threshold;
}

vector<bool> Snapshot::initMask(vector<int> actions_list){
	//mask=Signal([(ind in actions_list) for ind in xrange(self._SIZE)])
	vector<bool> result;
	for(int i=0;i<size;++i){
		bool flag=false;
		for(int j=0;j<actions_list.size();++j){
			if(i==actions_list[j]){
				flag=true;
				break;
			}
		}
		result.push_back(flag);
	}
	return result;
}

Snapshot::~Snapshot(){}

vector<vector<bool> > Snapshot::halucinate_all(vector<vector<int> > generalized_actions){
	vector<vector<bool> > result;
	for(int i=0;i<generalized_actions.size();++i){
		halucinate_GPU(generalized_actions[i]);
		result.push_back(this->getLoad());
	}
	return result;
}

vector<bool> Snapshot::halucinate(vector<int> action_list){
	halucinate_GPU(action_list);
	return this->getLoad();
}