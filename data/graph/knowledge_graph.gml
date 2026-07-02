graph [
  directed 1
  node [
    id 0
    label "distributed authorization framework"
    type "METHOD"
  ]
  node [
    id 1
    label "Istio service mesh"
    type "TOOL"
  ]
  node [
    id 2
    label "Kubernetes clusters"
    type "DATASET"
  ]
  node [
    id 3
    label "authorization speed"
    type "METRIC"
  ]
  node [
    id 4
    label "Envoy"
    type "TOOL"
  ]
  node [
    id 5
    label "service meshes"
    type "CONCEPT"
  ]
  node [
    id 6
    label "method to automate security policy enforcement"
    type "METHOD"
  ]
  node [
    id 7
    label "security policy enforcement"
    type "CONCEPT"
  ]
  node [
    id 8
    label "tool"
    type "TOOL"
  ]
  node [
    id 9
    label "zero-trust requirements"
    type "CONCEPT"
  ]
  node [
    id 10
    label "secure service mesh communication"
    type "CONCEPT"
  ]
  node [
    id 11
    label "cloud-native applications"
    type "CONCEPT"
  ]
  node [
    id 12
    label "unsupervised anomaly detection method"
    type "METHOD"
  ]
  node [
    id 13
    label "intrusion detection"
    type "CONCEPT"
  ]
  node [
    id 14
    label "autoencoders"
    type "METHOD"
  ]
  node [
    id 15
    label "anomaly scores"
    type "METRIC"
  ]
  node [
    id 16
    label "intrusion datasets"
    type "DATASET"
  ]
  node [
    id 17
    label "drift detection tool"
    type "TOOL"
  ]
  node [
    id 18
    label "system call telemetry"
    type "DATASET"
  ]
  node [
    id 19
    label "configuration drift"
    type "CONCEPT"
  ]
  node [
    id 20
    label "containerized environments"
    type "CONCEPT"
  ]
  node [
    id 21
    label "anomaly detection method"
    type "METHOD"
  ]
  node [
    id 22
    label "Envoy proxy"
    type "TOOL"
  ]
  node [
    id 23
    label "microservice traffic"
    type "DATASET"
  ]
  node [
    id 24
    label "Envoy JWT authorization rules"
    type "CONCEPT"
  ]
  node [
    id 25
    label "behavioral analysis"
    type "CONCEPT"
  ]
  node [
    id 26
    label "insider threats"
    type "CONCEPT"
  ]
  edge [
    source 0
    target 1
    relation "USES"
    confidence 1.0
    years "2021"
    year 2021
  ]
  edge [
    source 0
    target 2
    relation "EVALUATES_ON"
    confidence 0.9
    years "2021"
    year 2021
  ]
  edge [
    source 0
    target 3
    relation "IMPROVES"
    confidence 0.95
    years "2021"
    year 2021
  ]
  edge [
    source 4
    target 5
    relation "USES"
    confidence 0.85
    years "2021"
    year 2021
  ]
  edge [
    source 6
    target 7
    relation "PRODUCES"
    confidence 0.85
    years "2022"
    year 2022
  ]
  edge [
    source 8
    target 9
    relation "ADDRESSES"
    confidence 0.95
    years "2022"
    year 2022
  ]
  edge [
    source 8
    target 10
    relation "IMPROVES"
    confidence 0.9
    years "2022"
    year 2022
  ]
  edge [
    source 8
    target 11
    relation "APPLIED_TO"
    confidence 0.8
    years "2022"
    year 2022
  ]
  edge [
    source 8
    target 14
    relation "USES"
    confidence 1.0
    years "2020"
    year 2020
  ]
  edge [
    source 8
    target 15
    relation "PRODUCES"
    confidence 1.0
    years "2020"
    year 2020
  ]
  edge [
    source 8
    target 16
    relation "EVALUATES_ON"
    confidence 1.0
    years "2020"
    year 2020
  ]
  edge [
    source 12
    target 13
    relation "ADDRESSES"
    confidence 0.9
    years "2020"
    year 2020
  ]
  edge [
    source 17
    target 18
    relation "USES"
    confidence 0.95
    years "2019"
    year 2019
  ]
  edge [
    source 17
    target 19
    relation "ADDRESSES"
    confidence 0.9
    years "2019"
    year 2019
  ]
  edge [
    source 17
    target 20
    relation "APPLIED_TO"
    confidence 0.9
    years "2019"
    year 2019
  ]
  edge [
    source 21
    target 22
    relation "USES"
    confidence 0.9
    years "2025"
    year 2025
  ]
  edge [
    source 21
    target 23
    relation "EVALUATES_ON"
    confidence 0.95
    years "2025"
    year 2025
  ]
  edge [
    source 21
    target 24
    relation "LACKS"
    confidence 0.95
    years "2025"
    year 2025
  ]
  edge [
    source 25
    target 26
    relation "ADDRESSES"
    confidence 0.85
    years "2025"
    year 2025
  ]
]
