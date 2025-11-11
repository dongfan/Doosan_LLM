import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class FuelCommandListener(Node):
    def __init__(self):
        super().__init__('fuel_command_listener')

        # ✅ 서버에서 오는 명령 구독
        self.subscription = self.create_subscription(
            String,
            '/start_fuel',
            self.listener_callback,
            10)

        # ✅ fuel_task_manager 에게 전달할 퍼블리셔
        self.task_pub = self.create_publisher(String, '/fuel_task/start', 10)

        self.get_logger().info('💧 FuelCommandListener started — waiting for /start_fuel')

    def listener_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Invalid message: {e}")
            return

        self.get_logger().info(
            f"✅ Received fuel command — {data.get('fuelType')} / {data.get('amount')}원"
        )

        # ✅ fuel_task_manager로 메시지 전달
        task_msg = String()
        task_msg.data = json.dumps(data, ensure_ascii=False)
        self.task_pub.publish(task_msg)

        self.get_logger().info("📤 Published to /fuel_task/start")

def main(args=None):
    rclpy.init(args=args)
    node = FuelCommandListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
