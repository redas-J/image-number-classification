package mnist;

import fullyconnectednetwork.Network;
import fullyconnectednetwork.NetworkTools;
import trainset.TrainSet;

import java.io.File;

/**
 * Created by Luecx on 10.08.2017.
 */

public class Mnist {

    private final String path = "src/mnist/";

    private MnistImageFile image;
    private MnistLabelFile label;

    private int numberOfImagesUsed = 0;

    private final int printSteps = 0;

    public Mnist(){

        try {
            image = new MnistImageFile(path + "res/trainImage.idx3-ubyte", "rw");
            label = new MnistLabelFile(path + "res/trainLabel.idx1-ubyte", "rw");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public TrainSet createTrainSet(int numberOfImages) {

        TrainSet set = new TrainSet(28 * 28, 10);

        try {

            for(int i = 0; i < numberOfImages; i++) {

                double[] input = new double[28 * 28];
                double[] output = new double[10];

                output[label.readLabel()] = 1d;
                for(int j = 0; j < 28*28; j++){
                    input[j] = (double)image.read() / (double)256;
                }

                set.addData(input, output);
                image.next();
                label.next();


                if(printSteps != 0 && i % printSteps == 0) {
                    System.out.println("prepared: " + numberOfImagesUsed);
                }

                numberOfImagesUsed++;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return set;
    }

    public void trainData(Network net,TrainSet set, int epochs, int loops) {
        for(int e = 0; e < epochs; e++) {
            net.train(set, loops);

            if(printSteps != 0) System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>   "+ e + "   <<<<<<<<<<<<<<<<<<<<<<<<<<");
        }
    }

    public void testTrainSet(Network net, TrainSet set) {
        int correct = 0;
        for(int i = 0; i < set.size(); i++) {

            double highest = NetworkTools.indexOfHighestValue(net.calculate(set.getInput(i)));
            double actualHighest = NetworkTools.indexOfHighestValue(set.getOutput(i));
            if(highest == actualHighest) {
                correct++;
            }
            if(printSteps != 0 && i % printSteps == 0) {
                System.out.println(i + ": " + (double)correct / (double)(i + 1));
            }
        }

        double percentage = Math.round( (double)correct / (double)set.size() * (double)10000 ) / (double)100;

        System.out.println("Testing finished, RESULT: " + correct + " / " + set.size()+ "  -> " + percentage + " %");
    }
}
