/**
 * @projectName: CNNbyPureJava
 * @packageName: com.chi2liu.cnn.layerInterface
 * @fileName: Layer.java
 */
package com.chi2liu.cnn.layerInterface;

import java.util.function.Consumer;
import java.util.function.Function;

import com.chi2liu.cnn.data.DataSet;
import com.chi2liu.cnn.util.Size;

/**
 * @author: chi633 
 * @Email: cliu_whu@yeah.net
 * @date：2017年12月10日 下午3:50:08
 * @className: Layer.java
 * @version: 1.0
 */
public interface Layer {
	
	/**
	 * 获取此层的Size大小
	 * @return
	 * @date: 2017年12月10日
	 */
	Size getSize();
	
	/**
	 * 获取此层的输出map的个数
	 * @return
	 * @date: 2017年12月10日
	 */
	int getOutCount();
	
	
	/**
	 * 对此层的制定的一个记录的每一个输出进行操作
	 * 具体的操作定义在函数中
	 * @param recordIndex
	 * @param function
	 * @date: 2017年12月10日
	 */
	void forEachOutput(int recordIndex, Function<Integer, Consumer<double[][]>> function);
	
	/**
	 * 设置与之相连的下一层
	 * @param layer
	 * @date: 2017年12月10日
	 */
	void setNextLayer(Layer layer);
	
	/**
	 * 获取指定记录，指定的一个输出的errors
	 * @param recordIndex
	 * @param outIndex
	 * @return
	 * @date: 2017年12月10日
	 */
	double[][] getError(int recordIndex, Integer outIndex);
	
	/**
	 * 获取指定记录，指定的一个输出的结果
	 * @param recordIndex
	 * @param outIndex
	 * @return
	 * @date: 2017年12月10日
	 */
	double[][] getOut(int recordIndex, int outIndex);
	
	/**
	 * 对于一条记录进行前向传播
	 * @param record
	 * @date: 2017年12月10日
	 */
	void forward(DataSet.Record record);
	
	/**
	 * 对于一条记录进行反向传播
	 * @param record
	 * @return
	 * @date: 2017年12月10日
	 */
	boolean backPropagation(DataSet.Record record);

}
