#ifndef _AGENT_
#define _AGENT_

#include "Snapshot.h"
#include <string>

class Agent:public Snapshot{
public:
	Agent();
	Agent(double threshold);
	~Agent();
	void decide(string mode,vector<int> param1,string param2);
	vector<string> translate(vector<int> index_list);
	bool checkParam(vector<int> param);
	bool checkParam(string param);

	vector<string> getDecision();
	string getMessage();

private:
	vector<string> decision;
	string message;
};

#endif