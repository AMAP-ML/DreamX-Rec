# IntTravel Demo

This folder contains the public S/I/F implementation of IntTravel. Scenario/context features are encoded on S tokens, POI intention features remain on I tokens, and feedback features remain on F tokens. The model uses the paper configuration: a 3-layer task-guided HyperConnection encoder, task-specific selective gating, and DSFNet.

```bash
bash inttravel_demo/run.sh
```
