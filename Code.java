import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.gui.beans.TrainTestSplitMaker;
import weka.classifiers.trees.RandomForest;
import java.util.Random;
import weka.classifiers.evaluation.*;

public class Weka{
	
	public static void main(String[] args) throws Exception{
		
		DataSource dataset = new DataSource("C:/Users/Usuario/Downloads/iris.csv");
		Instances data = dataset.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		System.out.println(data);
		data.randomize(new Random(100));
		System.out.println(data);
		int tamTrain = (int) Math.round(150*0.7);
		int tamTest = 150 - tamTrain;
		Instances Train = new Instances(data, 0, tamTrain);
		Instances Test = new Instances(data, tamTrain, tamTest-1);
		RandomForest modelo = new RandomForest();
		modelo.buildClassifier(Train);
		Evaluation eval = new Evaluation(Test);
		eval.evaluateModel(modelo, Test);
		System.out.println(eval.toSummaryString());
		
	}
}
