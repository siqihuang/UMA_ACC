#ifndef _AGENT_
#define _AGENT_

#include "Snapshot.h"

class Agent:public Snapshot{
public:
	Agent();
	Agent(double threshold);
	~Agent();
	vector<vector<bool>> halucinate_all(vector<vector<int>> generalized_actions);
	vector<bool> halucinate(vector<int> action_list);
};

#endif