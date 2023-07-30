# InFlight_optimization

This repository contains the three notebooks that will be incorporated to the InFlight Optimization pipeline repository: https://github.com/8451LLC/inflight_offsite_media_optimization.git.

Three notebooks are included in this repository include feature extractions, training, and inference. 

The model data source includes Trade Desk REDS (Raw Event Data Stream) log data, media engagement features, embedding, and household purchase behaviors. 

A multi-task multi-output model, MTMO, is developed to calculate the bidding prices that will be sent to Trade Desk. MTMO is a model box, the best fluid model will be selected and deployed to production. The model currently takes in billions of rows of input data for model training. 

The expected IFO values include the new capabilities of personalization, improvement of attributable spend and lift, and cost saving. 

The challenges of IFO include extremely imbalanced sales conversion, key drivers for attributable spends such as social events not incorporated, no market information, only have won auctions logs, no lost auction logs, and trade Desk Data Governance that involves new contract and cost negotiation.

More details about IFO can be found here: 
https://confluence.kroger.com/confluence/display/8451SE/Inflight+Optimization#expand-Resources
