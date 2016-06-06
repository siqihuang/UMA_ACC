#include "Agent.h"

Agent::Agent():Snapshot(){}

Agent::Agent(double threshold):Snapshot(threshold){}

Agent::~Agent(){}

static int randInt(int n){
	return rand()%n;
}

void Agent::decide(string mode,vector<int> param1,string param2){//the decide function
	update_state_GPU(mode=="decide");
	if(mode=="execute"){
		if(checkParam(param1)){
			decision=translate(param1);
			message="Executing [";
			for(int i=0;i<param1.size();++i){
                                message+=(" "+std::to_string(param1[i]));
			}
			message+="]";
		}
		else{
			cout<<"Illegal input for execution by "<<name<<" --- Aborting!"<<endl;
			exit(0);
		}
	}
	else if(mode=="decide"){
		if(checkParam(param2)){
			vector<vector<bool> > responses;
			for(int i=0;i<generalized_actions.size();++i){
				responses.push_back(halucinate(generalized_actions[i]));
			}
			vector<int> best_responses;
			for(int i=0;i<responses.size();++i){
				if(responses[i][name_to_num[param2]]){
					best_responses.push_back(i);
				}
				//cout<<name_to_num[param2]<<endl;
			}
			//cout<<param2<<","<<responses[0].size()<<endl;
			
			if(!best_responses.empty()){
				decision=translate(generalized_actions[best_responses[randInt(best_responses.size())]]);
				message=param2+", ";
				for(int i=0;i<decision.size();++i) message+=(decision[i]+" ");
				
			}
			else{
				decision=translate(generalized_actions[randInt(generalized_actions.size())]);
				message=param2+", random";
			}
		}
		else{
			cout<<"Invalid decision criterion "<<param2<<" --- Aborting!"<<endl;
		}
	}
	else{
		cout<<"Invalid operation mode for agent "<<name<<" --- Aborting!"<<endl;
	}
}

vector<string> Agent::translate(vector<int> index_list){
	vector<string> name;
	for(int i=0;i<index_list.size();++i){
		name.push_back(sensors_names[index_list[i]]);
	}
	return name;
}

bool Agent::checkParam(vector<int> param){
	//exit(0);
	for(int i=0;i<generalized_actions.size();++i){
		if(param==generalized_actions[i]) return true;	
	}
	return false;
}

bool Agent::checkParam(string param){
	for(int i=0;i<evals_names.size();++i){
		if(param==evals_names[i]) return true;
	}
	return false;
}

vector<string> Agent::getDecision(){
	return decision;
}

string Agent::getMessage(){
	return message;
}
