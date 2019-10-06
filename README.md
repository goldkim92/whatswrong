# whatswrong
Hey SY and YW, to use the pretrained GoogLeNet, I built the model with GoogleNet by `4e` layer, followed by one conv layer and 3 dropout fully connected layer. 

## Run
To run the code
```
python googlenet_drop.py
```
<br>

In `googlenet_drop.py` file, 
- to freeze the model, you can set the argument `freeze` to `True`
- there are two images that are saved in `img` directory. The first one is `4e` layer (to see that the front layers are freezed), and the second one is the feature map by the output of the last conv layer.
