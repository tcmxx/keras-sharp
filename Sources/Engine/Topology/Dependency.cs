using System.Collections;
using System.Collections.Generic;
using UnityEngine;
namespace KerasSharp.Engine.Topology
{
    using System;
    using System.Runtime.Serialization;
    using System.Diagnostics;
    public abstract class Dependency : IDisposable
    {
            
        public abstract void Dispose();
        
    }

}
