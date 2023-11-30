# Schema of ZX env observation when batched into GraphTensor.
OBSERVATION_SCHEMA_ZX_final = """
# proto-file: //third_party/py/tensorflow_gnn/proto/graph_schema.proto
# proto-message: tensorflow_gnn.GraphSchema
context {
  features {
    key: "context_feat"
    value {
      dtype: DT_FLOAT
      shape { dim { size: 17 } }
    }
  }  
}
node_sets {
  key: "spiders"
  value {
    features {
      key: "color"
      value {
        dtype: DT_INT32
        shape { dim { size: 5 } }
      }
    }
    features {  
      key: "angle"
      value {
        dtype: DT_INT32
        shape { dim { size: 6 } }
      }
    }
    features {  
      key: "selected_node"
      value {
        dtype: DT_INT32
        shape { dim { size: 1 } }
      }
    }
  }
}
edge_sets {
  key: "edges"
  value {
    source: "spiders"
    target: "spiders"
    features {
        key: "selected_edges"
        value {
          dtype: DT_INT32
          shape { dim { size: 1 } }
        }
    }
  }
}

"""