/**
 * @projectName: CNNbyPureJava
 * @packageName: com.chi2liu.cnn.util
 * @fileName: Rectangle.java
 */
package com.chi2liu.cnn.util;

/**
 * @author: chi633 
 * @Email: cliu_whu@yeah.net
 * @date：2017年12月10日 下午3:29:15
 * @className: Rectangle.java
 * @version: 1.0
 */
public class Rectangle {
	private final double[][] values;

    public Rectangle(int width, int height) {
        values = new double[width][height];
    }

    public void set(int x, int y, double value) {
        values[x][y] = value;
    }
}
