### Experiment and Snapshot base classes

from multiprocessing import Pool
#import cPickle as nedosol
import numpy as np
from numpy.random import randint as rand
import UMA_NEW
acc=UMA_NEW.Agent()


N_CORES=8

### EXPERIMENT:
### Need new mechanism for adding measurables while recording dependencies
###

### MEASURABLES SHOULD BE A CLASS? 
### SIGNALS SHOULD BE A CLASS?

### SNAPSHOT:
### Stop using sensor names in computations
### Parallelize weight computation

###
### Shortcuts to numpy Boolean Logic functions
###

def negate(x):
      return np.logical_not(x)

def conjunction(x,y):
      return np.logical_and(x,y)

def disjunction(x,y):
      return np.logical_or(x,y)

def symmetric(x,y):
      return np.logical_xor(x,y)

def alltrue(n):
      return np.array([True for x in xrange(n)])

def allfalse(n):
      return np.array([False for x in xrange(n)])

###
### Name-handling functions
###

def name_comp(name):
      ### return the name of the complementary sensor
      return name+'*' if name[-1:]!='*' else name[:-1] 

def name_invert(names):
      ### return the set of complemented names in the list/set names
      return set(comp(name) for name in names)

def wedge(name0,name1):
      ### conjunction of two names
      return '('+str(name0)+'^'+str(name1)+')'

###
### Normalizing functions
###

def simplex_normalize(probs):
      s=sum([x for x in probs if x>0])
      return [((x+0.)/(s+0.) if x>0. else 0.) for x in probs]

###
### DATA STRUCTURES
###

class Signal(object):
      def __init__(self,value):
            if len(value)%2==0:
                  self._VAL=np.array(value,dtype=bool)
            else:
                  raise Exception('Objects of class Signal must have even length -- Aborting!\n')

      def __repr__(self):
            print self._VAL

      ### set the signal
      def set(self,ind,value):
            self._VAL[ind]=value

      def value(self,ind):
            return self._VAL[ind]

      def extend(self,value):
            if len(value)%2==0 and type(value)==type(self._VAL):
                  self._VAL=np.array(list(self._VAL)+list(value))
            else:
                  raise Exception("Cannot extend a signal by something that's not a signal -- Aborting!\n")
                  
      ### negating a partial signal
      def star(self):
            return Signal([(self._VAL[i+1] if i%2==0 else self._VAL[i-1]) for i in xrange(len(self._VAL))])
      
      ### full complement of a signal
      def negate(self):
            return Signal(negate(self._VAL))

      ### subtracting Signal "other" from Signal "self"
      def subtract(self,other):
            return Signal(conjunction(self._VAL,negate(other._VAL)))

      def add(self,other):
            return Signal(disjunction(self._VAL,other._VAL))

      def intersect(self,other):
            return Signal(conjunction(self._VAL,other._VAL))


class Measurable(object):
      ### initialize a measurable
      def __init__(self,name,experiment,value,definition=None):

            ### name will remain unchanged
            self._NAME=str(name)

            ### experiment will remain unchanged
            self._EXPERIMENT=experiment

            ### initial value is a stack containing no more than $experiment._DEPTH+1$ values
            self._VAL=value

            ### a function of the experiment state (the latter is a dictionary indexed by [all] measurables)
            self._DEFN=definition
      
      def __repr__(self):
            return str(self._VAL)

      def val(self):
            return self._VAL

      ### Arbitrarily set the current value of a measurable:
      ### (pushes preceding values down the value stack)
      def set(self,value):
            self._VAL=[value]+list(self.val())[:self._EXPERIMENT._DEPTH]
            return self._VAL

 

class Experiment(object):
      def __init__(self,depth):
            ### depth of data recording in the experiment:
            self._DEPTH=depth
            ### list of agents in this experiment:
            self._AGENTS=[]

            ### measurables corresponding to binary control signals
            self._CONTROL=[]
            ### list of measurables ordered to accommodate dependencies during the updating process:
            self._MEASURABLES=[]
            ### name-based look-up table (dictionary) for the unified list of  measurables, including control signals:
            self._MNAMES={}

      ### represent an experiment
      def __repr__(self):
            intro="Experiment of depth "+str(self._DEPTH)+" including the agents "+str([agent._NAME for agent in self._AGENTS])+".\nThe current state is:\n"
            state_data=""
            ### SWITCH TO _CONTROLS and _MEASURABLES HERE:
            for meas in self._CONTROL:
                  state_data+=meas._NAME+":  "+str(meas.val())+"\n"
            for meas in self._MEASURABLES:
                  state_data+=meas._NAME+":  "+str(meas.val())+"\n"

            return intro+state_data

      ### initialize an agent
      def add_agent(self,name,threshold):
            new_agent=Agent(name,self,threshold)
            self._AGENTS.append(new_agent)
            return new_agent


      ### initialize a measurable -- measurables are observable quantities 
      ### of the experiment, init is a $self._DEPTH+1$-dimensional vector of
      ### values of the measurable $name$.
      ###
      ### returns the new measurable
      def add_measurable(self,name,init,definition=None):
            if name in self._MNAMES:
                  raise Exception("The name ["+name+"] is already in use as the name of a measurable in this experiment -- Aborting!\n\n")
            else:
                  # construct the new measurable
                  new_measurable=Measurable(name,self,init,definition)
                  #the case of a control signal
                  if definition==None:
                        self._CONTROL.append(new_measurable)
                  # the case of a dependent measurable
                  else:
                        self._MEASURABLES.append(new_measurable)

                  # add the new measurable to related dictionaries
                  self._MNAMES[name]=new_measurable

                  return new_measurable

      ### query the state of the experiment:
      def state(self,name,delta=0):
            return self._MNAMES[name].val()[delta:]

      def state_all(self):
            return {name:self._MNAMES[name].val() for name in self._MNAMES.keys()}

      ### SIMULATION ONLY: update the state of the experiment given a choice 
      ### of actions, $action_signal$.
      ###
      ### we assume $action_signal$ is a boolean list corresponding to 
      ### self._CONTROL
      def update_state(self,action_signal):
            last_state={name:self._MNAMES[name].val() for name in self._MNAMES}
            for ind,meas in enumerate(self._CONTROL):
                  last_state[meas._NAME]=meas.set(action_signal[ind])
                  #print meas._NAME,meas.val()
                  
            for meas in self._MEASURABLES:
                  last_state[meas._NAME]=meas.set(meas._DEFN(last_state))
                  #print meas._NAME,meas.val()

      ### ADD A SENSOR TO THE EXPERIMENT
      ###
      def new_sensor(self,agent_list,name,definition=None):
            # SETTING UP NEW SENSOR IN THE EXPERIMENT:
            # compute the initial values of the new measurable:
            # put 'False' in both $name$ and $name*$ over full depth of experiment
            new_meas=self.add_measurable(name,allfalse(1+self._DEPTH),definition)

            # $definition==None$ is used to designate action sensors
            new_meas_comp=self.add_measurable(name_comp(name),alltrue(1+self._DEPTH) if definition==None else allfalse(1+self._DEPTH),None if definition==None else lambda state: negate(definition(state)))

            ### add the sensor to the agents
            for agent in agent_list:
                  agent.add_sensor(name,new_meas,new_meas_comp,definition==None)

            ### return the new pair of measurables
            return new_meas,new_meas_comp

      ### FORM A DELAYED CONJUNCTION
      ### (name1 is the delayed sensor)
      def twedge(self,agent_list,name0,name1):
            if name0 in self._MNAMES and name1 in self._MNAMES:
                  def newdefn(state):
                        return conjunction(state[name0][0],state[name1][1])
                  self.new_sensor(agent_list,wedge(name0,name1),newdefn)
                  for agent in agent_list:
                        agent.set_context(name0,name1,wedge(name0,name1))
                        #agent._CONTEXT[(agent._NAME_TO_NUM[name0],agent._NAME_TO_NUM[name1])]=agent._NAME_TO_NUM[wedge(name0,name1)]
            else:
                  raise('One of the provided component names is undefined in agent '+str(self._NAME)+' -- Aborting.\n')
 
                              
      ### ONE TICK OF THE CLOCK: 
      ### have agents decide what to do, then collect their decisions and update the measurables.
      def tick(self,mode,param):
            decision=[]
            for agent in self._AGENTS:
		  #------------------THIS IS THE GPU PART---------------------#
                  #dec,message=agent.decide(mode,param)

		  #the following is for the last command
		  #data input
		  for ind in xrange(agent._SIZE):
            	        agent._OBSERVE.set(ind,agent._SENSORS[ind].val()[0])
		  acc.setSignal(agent._OBSERVE._VAL.tolist())

		  #based on different type of param, I use two parameter, so the type judgement here is necessary
		  if type(param) is list:
			acc.decide(mode,param,'')
		  else:
			acc.decide(mode,[],param)
		  #after the GPU part is done, two variables are get from C++ and I cannot use return value because C++ can only return one value
		  dec=acc.getDecision()
		  message=acc.getMessage()
		  #------------------THIS IS THE GPU PART---------------------#

                  decision.extend(dec)

            self.update_state([(meas._NAME in decision) for meas in self._CONTROL])
            return message


class Snapshot(object):
      ### initialize an empty snapshot with a list of sensors and a learning threshold
      ###
      def __init__(self,name,experiment,threshold):
            self._NAME=str(name) ### a string naming the snapshot
            self._EXPERIMENT=experiment ### the experiment observed by the snapshot
            self._SIZE=0 ### snapshot size is always even
            ### an ordered list of Boolean measurables used by the agent:
            self._SENSORS=[] 
            ### LOOKUP TABLES TO SAVE TIME
            ### a list of the actions available to the agent
            ### (indices in the self._SENSORS array)
            self._ACTIONS=[]
            self._GENERALIZED_ACTIONS=[[]]
            ### a list of available evaluator sensors:
            ### (establishes the agent's priorities)
            self._EVALS=[]
            ### a dictionary of sensor indices
            self._NAME_TO_NUM={} # $name:number$ pairs
            self._NAME_TO_SENS={} # $name:sensor$ pairs
            
            ### Boolean vectors ordered according to self._SENSORS
            ### raw observation:
            self._OBSERVE=Signal(np.array([],dtype=np.bool))
            ### current state representation:
            self._CURRENT=Signal(np.array([],dtype=np.bool))

            ### poc set learning machinery:
            ### learning thresholds matrix
            self._THRESHOLDS=np.array([[threshold]])
            ### snapshot weight matrix
            self._WEIGHTS=np.array([[0.]])
            ### snapshot graph matrix
            self._DIR=np.array([[False]],dtype=np.bool)

            ### context is a dictionary of the form (i,j):k indicating that sensor number k was constructed as twedge(i,j).
            self._CONTEXT={}
   
                  
      def __repr__(self):
            return 'The snapshot '+str(self._NAME)+' has '+str(self._SIZE/2)+' sensors:\n\n'+str([meas._NAME for ind,meas in enumerate(self._SENSORS) if ind%2==0])+'\nout of which the following are actions:\n'+str([self._SENSORS[ind]._NAME for ind in self._ACTIONS])+'\n\n'
      
      ### adding a sensor to the agent
      ###
      def add_sensor(self,name,new_meas,new_meas_comp,actionQ):
            ### EXTENDING THE SNAPSHOT:
            self._SIZE+=2
            # adding new sensors to lists
            self._SENSORS.extend([new_meas,new_meas_comp])
            if actionQ:
                  self._ACTIONS.append(self._SIZE-2)
                  temp_list=self._GENERALIZED_ACTIONS[:]
                  for item in self._GENERALIZED_ACTIONS:
                        temp_list.extend([item+[self._SIZE-2],item+[self._SIZE-1]])
                        temp_list.remove(item)
                  self._GENERALIZED_ACTIONS=temp_list
            # extending lookup tables
            self._NAME_TO_SENS[name]=new_meas
            self._NAME_TO_SENS[name+'*']=new_meas_comp
            self._NAME_TO_NUM[name]=self._SIZE-2
            self._NAME_TO_NUM[name+'*']=self._SIZE-1
            ### update current values of sensor according to definition in both the experiment and the snapshot:
            self._OBSERVE.extend(np.array([self._EXPERIMENT.state(name)[0],self._EXPERIMENT.state(name+'*')[0]]))
            self._CURRENT.extend(np.array([False,False]))
            ### preparing weight, threshold and direction matrices:
            self._WEIGHTS=np.array([[self._WEIGHTS[0][0] for col in range(self._SIZE)] for row in range(self._SIZE)])
            self._THRESHOLDS=np.array([[self._THRESHOLDS[0][0] for col in range(self._SIZE)] for row in range(self._SIZE)])
            self._DIR=np.array([[False for col in range(self._SIZE)] for row in range(self._SIZE)],dtype=np.bool)


      def add_eval(self,name):
            # any non-action sensor may serve as an evaluation sensor
            if name in self._NAME_TO_NUM and self._NAME_TO_NUM[name] not in self._ACTIONS:
                  self._EVALS.append(name)

      def set_context(self,name0,name1,name_tc):
            self._CONTEXT[(self._NAME_TO_NUM[name0],self._NAME_TO_NUM[name1])]=self._NAME_TO_NUM[name_tc]

      ### MAKE OBSERVATION AND UPDATE INTERNAL CURRENT STATE
      def update_state(self,mode='decide'):
            ### read the values of all sensors into self._OBSERVE
            for ind in xrange(self._SIZE):
                  self._OBSERVE.set(ind,self._SENSORS[ind].val()[0])
            ### update the weights and poc graph
            self.update_weights()
	    #acc.setSignal(self._OBSERVE._VAL.tolist())
	    #self._WEIGHTS=np.array(acc.update_weights_GPU(self._WEIGHTS.tolist()))
            if mode!='execute':
                  self.orient_all()
            ### propagate raw observation to obtain self._CURRENT
            self._CURRENT=self.propagate(self._OBSERVE,Signal(allfalse(self._SIZE)))

      ### UPDATE WEIGHTS FROM RAW OBSERVATION
      def update_weights(self):
            for nrow in xrange(self._SIZE):
                  for ncol in xrange(self._SIZE):
                        self._WEIGHTS[nrow][ncol]+=self._OBSERVE._VAL[nrow]*self._OBSERVE._VAL[ncol]
   
                              
      ### test for the relation row<col where row,col are sensor names
      ###
      def implies(self,row,col):
            ### allow testing a pair of sensors by name
            if type(row)==type("string"):
                  row=self._NAME_TO_NUM[row]
                  col=self._NAME_TO_NUM[col]

            ### applying the involution to indices (rather than names)
            compi=lambda x: x+1 if x%2==0 else x-1

            ### row and col are now indices in range(self._SIZE)
            rc=0.+self._WEIGHTS[row][col]
            r_c=0.+self._WEIGHTS[compi(row)][col]
            rc_=0.+self._WEIGHTS[row][compi(col)]
            r_c_=0.+self._WEIGHTS[compi(row)][compi(col)]
            epsilon=(0.+rc+r_c+rc_+r_c_)*self._THRESHOLDS[row][col]
            ### return True if rc_ is negligibly small compared to other sides of the square
            return (rc_<np.amin([epsilon,rc,r_c,r_c_]))

      def equivalent(self,row,col):
            ### allow testing a pair of sensors by name:
            if type(row)==type("string"):
                  row=self._NAME_TO_NUM[row]
                  col=self._NAME_TO_NUM[col]

            ### apply involution to indices:
            compi=lambda x: x+1 if x%2==0 else x-1

            ### row and col are indices in range(self._SIZE)
            rc=0.+self._WEIGHTS[row][col]
            r_c=0.+self._WEIGHTS[compi(row)][col]
            rc_=0.+self._WEIGHTS[row][compi(col)]
            r_c_=0.+self._WEIGHTS[compi(row)][compi(col)]
            epsilon=(rc+r_c+rc_+r_c_)*self._THRESHOLDS[row][col]
            ### return True if rc_=r_c=0 and rc,r_c_>epsilon
            return (rc_==0 and r_c==0)# and epsilon<0.+rc and epsilon<0.+r_c_)
            
      
      ### recalculate the poc set structure self._DIR
      ###
      def orient_all(self):
            ### go over all squares...
            map(self.orient_square,[(x,y) for x in xrange(0,self._SIZE,2) for y in xrange(0,x,2)])


      ### orienting a single square
      def orient_square(self,(x,y)):
            compi=lambda x: x+1 if x%2==0 else x-1
            # wipe previous orientation
            self._DIR[x][y]=False
            self._DIR[x][compi(y)]=False
            self._DIR[compi(x)][y]=False
            self._DIR[compi(x)][compi(y)]=False
            self._DIR[y][x]=False
            self._DIR[compi(y)][x]=False
            self._DIR[y][compi(x)]=False
            self._DIR[compi(y)][compi(x)]=False

            # assuming x,y are even
            square_is_oriented=0
            for i in [0,1]:
                  for j in [0,1]:
                        sx=x+i
                        sy=y+j
                        if square_is_oriented==0:
                              if self.implies(sy,sx):
                                    self._DIR[sy][sx]=True
                                    self._DIR[compi(sx)][compi(sy)]=True
                                    self._DIR[sx][sy]=False
                                    self._DIR[compi(sy)][compi(sx)]=False
                                    self._DIR[sx][compi(sy)]=False
                                    self._DIR[compi(sy)][sx]=False
                                    self._DIR[sy][compi(sx)]=False
                                    self._DIR[compi(sx)][sy]=False
                                    square_is_oriented=1
                              if self.equivalent(sy,sx):
                                    self._DIR[sy][sx]=True
                                    self._DIR[sx][sy]=True
                                    self._DIR[compi(sx)][compi(sy)]=True
                                    self._DIR[compi(sy)][compi(sx)]=True
                                    self._DIR[sx][compi(sy)]=False
                                    self._DIR[compi(sy)][sx]=False
                                    self._DIR[sy][compi(sx)]=False
                                    self._DIR[compi(sx)][sy]=False
                                    square_is_oriented=1

      ### Signal to Names
      def Signal_to_Names(self,signal):
            return [sens._NAME for ind,sens in enumerate(self._SENSORS) if signal.value(ind)]

      ### String to signal
      def Names_to_Signal(self,names):
            return Signal([(sens._NAME in names) for sens in self._SENSORS])

      ### UPWARD CLOSURE OF A SIGNAL:
      def up(self,input):
            visited=allfalse(self._SIZE)

            def dfs(ind):
                  if not visited[ind]:
                        visited[ind]=True
                        map(dfs,[col for col in xrange(self._SIZE) if self._DIR[ind][col]])

            if type(input)!=Signal: #assumes signal is a list of sensor names
                  signal=self.Names_to_Signal(input)
            else:
                  signal=input

            map(dfs,[ind for ind in xrange(self._SIZE) if signal._VAL[ind]])

            return Signal(visited)


      ### PROPAGATION
      ###
      ### (takes a True/False signal to propagate over a state)
      def propagate(self,signal,load):
            load=self.up(load)
            mask_pos=self.up(signal)
            return (load.add(mask_pos)).subtract(mask_pos.star())
            

      ### simulate the effect of a generalized action (input is a complete *-selection over $self._ACTIONS$, taken from $self._GENERALIZED_ACTIONS$, represented with sensor indices)
      def halucinate(self,actions_list):
            # form a mask signal to hold the output while recording the designated actions on this mask:
            mask=Signal([(ind in actions_list) for ind in xrange(self._SIZE)])

            # establish context pairs (actions for which there is a delayed conjunction on record, relevent at the current state)
            relevant_pairs=[(act,ind) for act in actions_list for ind in xrange(self._SIZE) if (act,ind) in self._CONTEXT and self._CURRENT.value(ind)]
            # record the corresponding delayed conjunctions in the mask
            map(mask.set,[self._CONTEXT[i,j] for i,j in relevant_pairs],[True for i,j in relevant_pairs])

            return self.propagate(mask,self._CURRENT)


class Agent(Snapshot):
      ### LATER MOVE ALL SNAPSHOT UPDATING TO THIS CLASS
      
      ### DECISION-MAKING MECHANISM
      def decide(self,mode,param):
            ### update the snapshot

	    self.update_state(mode)
	    
	    #for ind in xrange(self._SIZE):
            	  #self._OBSERVE.set(ind,self._SENSORS[ind].val()[0])
	    #acc.setSignal(self._OBSERVE._VAL.tolist())
	    #acc.update_state_GPU(mode=='decide')
	    #self._CURRENT=Signal(np.array(acc.getCurrent()))
	    #self._DIR=np.array(acc.getDir())

            # translate indices into names for output to experiment
            translate=lambda index_list: [self._SENSORS[ind]._NAME for ind in index_list]
	    
            if mode=='execute':
                  if param in self._GENERALIZED_ACTIONS:
                        decision=translate(param)
                        message='Executing '+str(param)
                  else:
                        raise('Illegal input for execution by '+str(self._NAME)+' --- Aborting!\n\n')
            elif mode=='decide':
                  if param in self._EVALS:
                        responses=map(self.halucinate,self._GENERALIZED_ACTIONS)
			#responses=[]
			#for actionlist in self._GENERALIZED_ACTIONS:
			      #responses.append(Signal(np.array(acc.halucinate(actionlist))))
            
                        # compute the response (if any) to the motivational signal
                        best_responses=[]

                        for ind in xrange(len(responses)):
                              if responses[ind].value(self._NAME_TO_NUM[param]):
                                    best_responses.append(ind)

                                    
                        if best_responses!=[]:
                              decision=translate(self._GENERALIZED_ACTIONS[best_responses[rand(len(best_responses))]])
                              message=param+', '+str(decision)

                        else:
                              decision=translate(self._GENERALIZED_ACTIONS[rand(len(self._GENERALIZED_ACTIONS))])
                              message=param+', random'
                  else:
                        raise('Invalid decision criterion '+str(param)+' --- Aborting!\n\n')
            else:
                  raise('Invalid operation mode for agent '+str(self._NAME)+' --- Aborting!\n\n')

            return decision,message

