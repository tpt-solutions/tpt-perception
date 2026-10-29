# tpt-percept-ros

Converts between `tpt-perception` clouds and the ROS 2 `sensor_msgs/PointCloud2`
byte layout. Pure Rust with no ROS dependency, so it is tested on any platform.

```rust
use tpt_percept_ros::{from_point_cloud2, to_point_cloud2};

let (layout, bytes) = to_point_cloud2(&scan);
let back = from_point_cloud2(&layout, &bytes, false)?;
```

Supported: little-endian buffers, all ROS datatypes, `x y z` plus optional
`intensity`. Big-endian messages are rejected. A `rclrs` node that subscribes to
and publishes these messages is not included, since building it needs a ROS 2
installation.
