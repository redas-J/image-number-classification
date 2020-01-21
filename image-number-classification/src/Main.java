import fullyconnectednetwork.Network;
import mnist.Mnist;
import trainset.TrainSet;

/**
 * @author Redas Jatkauskas
 */

public class Main {

    public static void main(String[] args) {

        if(args.length < 3){
            System.err.println("Too few arguments");
            return;
        }
        int numberOfImages = Integer.parseInt(args[0]);
        int epochs = Integer.parseInt(args[1]);
        int loops = Integer.parseInt(args[2]);
        int[] network_layer_sizes = new int[args.length - 3];
        for (int i = 0; i < args.length - 3; i++) {
            network_layer_sizes[i] = Integer.parseInt(args[i+3]);
        }

        Network network = new Network(network_layer_sizes);
        Mnist mnist = new Mnist();

        TrainSet trainSet = mnist.createTrainSet(numberOfImages);
        mnist.trainData(network, trainSet, epochs, loops);

        TrainSet testSet = mnist.createTrainSet( numberOfImages);
        mnist.testTrainSet(network, testSet);
    }

}
