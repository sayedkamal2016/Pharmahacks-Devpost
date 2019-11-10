import zmq
import json

class MLConnector():
	
	context = None #For zmq you have to define this otherwise it wont start
	socket = None

	def _init_(self):
		self.context = zmq.Context()
		self.socket = self.context.socket(zmq.REP)
		self.socket.bind("tcp://*:5555")
		

	def sendResults(self, result):
		print("sending results")
		self.socket.send(result)
	
	def recvResults(self):
		message = self.socket.recv()
		print(message)
		return message


if __name__ == '__main__':
	#Has to run 
	while(True):
		connector = MLConnector()
		connector._init_() #Not normal
		connector.recvResults()
		connector.sendResults(result=b"Hello YES")
