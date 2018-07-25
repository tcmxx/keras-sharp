using KerasSharp;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using KerasSharp.Engine.Topology;
using static KerasSharp.Backends.Current;
using System;
using Accord.Math;
using System.Linq;

public class Concat : Merge {

    public int axis;

    public Concat(int axis)
    {
        this.axis = axis;

        //this.supports_masking = true;
    }


    protected override List<Tensor> InnerCall(List<Tensor> inputs, List<Tensor> mask = null, bool? training = null)
    {
        return new List<Tensor>() { K.concat(inputs, axis) };
    }

    public override List<int?[]> compute_output_shape(List<int?[]> input_shapes)
    {
        if (input_shapes.Count <= 1)
            throw new Exception("Expected a at least one input.");

        int?[] input_shape = input_shapes[0];
        if (axis >= input_shape.Length || axis < 0)
            throw new Exception("Concat axis not withing the input shape rank range.");
        if (!input_shape[axis].HasValue)
            throw new Exception("Can not concat unknow dimension.");

        int concatedLength = 0;

        for(int i = 0; i < input_shapes.Count; ++i)
        {
            if(input_shapes[i].Length != input_shape.Length)
            {
                throw new Exception("Input dimensions not match. Can not concat.");
            }
            for(int j = 0; j < input_shape.Length; ++j)
            {
                if(input_shape[j].HasValue && j != axis &&(!input_shapes[i][j].HasValue || input_shapes[i][j].Value != input_shape[j].Value))
                {
                    throw new Exception("Input dimensions not match. Can not concat.");
                }else if(j == axis)
                {
                    if(!input_shapes[i][j].HasValue )
                        throw new Exception("Concat axis can not be unknow.");
                    concatedLength += input_shapes[i][j].Value;
                }
            }
        }

        int?[] output_shape = input_shape.Copy();
        output_shape.Set(index: axis, value: concatedLength);
        return new[] { output_shape }.ToList();
    }
}
