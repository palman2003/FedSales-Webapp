# FedSales - Empowering Demand with Federated Learning
A Web based application for Demand Forecasting using Federated Learning

The flwr_server.py must be initialized first in cmd and the server requires a minimum of two clients.
initialize 2 clients by seperately opening two different terminals in cmd
the server uses a FedAVG strategy to fit the learning data from the clients

the webapp uses the model that has learnt in a federated manner from the clients and combines all the weights that is stored in .hdf5 file

the webapp uses flask for backend and HTML,CSS,JavaScript for the frontend
Federated Learning was implemented using Flower and Tensorflow(LSTM) model.
