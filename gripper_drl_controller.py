import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import DrlStart
import textwrap

DRL_BASE_CODE = """
g_slaveid = 0
flag = 0
def modbus_set_slaveid(slaveid):
    global g_slaveid
    g_slaveid = slaveid
def modbus_fc06(address, value):
    global g_slaveid
    data = (g_slaveid).to_bytes(1, byteorder='big')
    data += (6).to_bytes(1, byteorder='big')
    data += (address).to_bytes(2, byteorder='big')
    data += (value).to_bytes(2, byteorder='big')
    return modbus_send_make(data)
def modbus_fc16(startaddress, cnt, valuelist):
    global g_slaveid
    data = (g_slaveid).to_bytes(1, byteorder='big')
    data += (16).to_bytes(1, byteorder='big')
    data += (startaddress).to_bytes(2, byteorder='big')
    data += (cnt).to_bytes(2, byteorder='big')
    data += (2 * cnt).to_bytes(1, byteorder='big')
    for i in range(0, cnt):
        data += (valuelist[i]).to_bytes(2, byteorder='big')
    return modbus_send_make(data)
def recv_check():
    size, val = flange_serial_read(0.1)
    if size > 0:
        return True, val
    else:
        tp_log("CRC Check Fail")
        return False, val
def gripper_move(stroke):
    flange_serial_write(modbus_fc16(282, 2, [stroke, 0]))
    wait(1.0) # 물리적 동작 시간을 충분히 기다려줍니다.
#현재 위치 읽기 함수
def gripper_get_position():
    flange_serial_write(modbus_fc03(290, 2))
    flag, val = recv_check()
    if flag == True:
        if val[1] == 3 and val[2] == 4:
            reg_low_word = (val[3] << 8) | val[4]
            reg_high_word = (val[5] << 8) | val[6]
            present_position = (reg_high_word << 16) | reg_low_word
            return True, present_position
        else:
            tp_log("gripper_get_position: Invalid response frame")
            return False, 0
    tp_log("gripper_get_position: Failed to read")
    return False, 0

# ---- init serial & torque/current ----
while True:
    flange_serial_open(
        baudrate=57600,
        bytesize=DR_EIGHTBITS,
        parity=DR_PARITY_NONE,
        stopbits=DR_STOPBITS_ONE,
    )

    modbus_set_slaveid(1)

    # 256(40257) Torque enable
    # 275(40276) Goal Current
    # 282(40283) Goal Position

    flange_serial_write(modbus_fc06(256, 1))   # torque enable
    flag, val = recv_check()

    flange_serial_write(modbus_fc06(275, 400)) # goal current
    flag, val = recv_check()

    if flag is True:
        break

    flange_serial_close()
"""

class GripperController:
    def __init__(self, node: Node, namespace: str = "dsr01", robot_system: int = 0):
        self.node = node
        self.robot_system = robot_system
        self.cli = self.node.create_client(DrlStart, f"/{namespace}/drl/drl_start")

        self.node.get_logger().info(f"Waiting for service /{namespace}/drl/drl_start...")
        while not self.cli.wait_for_service(timeout_sec=2.0):
            self.node.get_logger().info("Service not available, waiting again...")
        self.node.get_logger().info("GripperController is ready.")

    def _send_drl_script(self, code: str) -> bool:
        req = DrlStart.Request()
        req.robot_system = self.robot_system
        req.code = code
        future = self.cli.call_async(req)
        
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
        if future.result() is not None:
            return bool(future.result().success)
        else:
            self.node.get_logger().warn(f"Service call failed: {future.exception()}")
            return False

    def initialize(self) -> bool:
        self.node.get_logger().info("Initializing gripper connection...")
        task_code = textwrap.dedent("""
            flange_serial_open(baudrate=57600, bytesize=DR_EIGHTBITS, parity=DR_PARITY_NONE, stopbits=DR_STOPBITS_ONE)
            modbus_set_slaveid(1)
            flange_serial_write(modbus_fc06(256, 1))
            recv_check()
            flange_serial_write(modbus_fc06(275, 400))
            recv_check()
        """)
        init_script = f"{DRL_BASE_CODE}\n{task_code}"
        success = self._send_drl_script(init_script)
        if success:
            self.node.get_logger().info("Gripper connection initialized successfully.")
        else:
            self.node.get_logger().error("Failed to initialize gripper connection.")
        return success

    def move(self, stroke: int, wait_time: float = 2.0) -> bool:
        self.node.get_logger().info(f"Moving gripper to stroke: {stroke}")
        task_code = textwrap.dedent(f"""
            gripper_move({stroke})
            wait({wait_time})  # ✅ DRL 내부에서 완료 대기
        """)
        move_script = f"{DRL_BASE_CODE}\n{task_code}"
        success = self._send_drl_script(move_script)
        if success:
            self.node.get_logger().info(f"✅ Gripper move({stroke}) finished after wait({wait_time}s).")
        else:
            self.node.get_logger().warn(f"❌ Failed to send gripper move command ({stroke}).")

        return success

    def get_position(self) -> int:
        self.node.get_logger().info("Getting gripper position...")
        task_code = textwrap.dedent("""
            success, position = gripper_get_position()
            if success:
                tp_log(f"Gripper position: {position}")
            else:
                tp_log("Failed to get gripper position.")
        """)
        position_script = f"{DRL_BASE_CODE}\n{task_code}"
        success = self._send_drl_script(position_script)
        if success:
            self.node.get_logger().info("Gripper position retrieval command sent successfully.")
        else:
            self.node.get_logger().error("Failed to send gripper position retrieval command.")
        return -1  # Actual position retrieval would require more complex handling

    def terminate(self) -> bool:
        self.node.get_logger().info("Terminating gripper connection...")
        terminate_script = "flange_serial_close()"
        success = self._send_drl_script(terminate_script)
        if success:
            self.node.get_logger().info("Gripper connection terminated successfully.")
        else:
            self.node.get_logger().error("Failed to terminate gripper connection.")
        return success
