# Newton Quest Teleop 启动流程

## 1. 可选：准备 sim-screen 视频设备

默认 Docker 启动会保留 Quest 里的 sim-screen XR plane / 手骨架 overlay，所以需要
`/dev/video44`。遥操输入仍然直接来自 Quest/OpenXR，不从 overlay JSONL 读手部样本。

```bash
sudo modprobe -r v4l2loopback

sudo modprobe v4l2loopback video_nr=44 card_label=teleop_sim_screen exclusive_caps=1 max_buffers=2 max_width=1920 max_height=1080
```

确认：

```bash
ls -l /dev/video44
```

非docker启动，需要在newton conda环境中启动

scripts/run_newton_vr_prereqs_object.sh --display :0


## 2. Docker 一条命令启动

```bash
cd ~/project/newton
newgrp docker
DISPLAY=:0 docker/run_vr_stack.sh
```

这条默认会启动 sim-screen XR plane / 手骨架 overlay，同时 Newton 直接从 Quest/OpenXR
读取手部输入。这和远端 Teleop 链路一致：显示链路是旁路，遥操主循环不读 overlay 日志。

保留功能前提下可以调这些性能参数：

```bash
# 降低 X11 -> v4l2loopback 的 CPU 拷贝压力
NEWTON_VR_CAPTURE_FPS=15 DISPLAY=:0 docker/run_vr_stack.sh

# 指定 VR 屏幕捕获尺寸，默认会自动夹到实际 X11 display 尺寸
NEWTON_VR_CAPTURE_SIZE=1024x768 DISPLAY=:0 docker/run_vr_stack.sh

# 双 GPU 机器上可试：把 camera_streamer 容器限制到 GPU1，Newton scene 继续用 cuda:0
NEWTON_CAMERA_STREAMER_VISIBLE_DEVICES=1 DISPLAY=:0 docker/run_vr_stack.sh
```

如果只想临时排查遥操输入、不要 VR 屏幕和手骨架 overlay：

```bash
DISPLAY=:0 docker/run_vr_stack.sh --skip-vr-output
```

只有调试旧链路时，才让 Newton 从 overlay JSONL 读取手部样本：

```bash
DISPLAY=:0 docker/run_vr_stack.sh --with-vr-output --teleop-input-source overlay-log
```

不要默认用 `sudo` 启动这条命令；如果确实用 `sudo DISPLAY=:0 docker/run_vr_stack.sh`，
脚本也会自动挂载原用户 home 下的 `.cache` 和 `.cloudxr`。

启动成功后终端应看到：

```text
CloudXR web client is serving https://127.0.0.1:8443/
Quest web page: https://192.168.8.100:8443/
OpenXR runtime found but no active Quest session yet; retrying...
```

这表示主机已经在等待 Quest 连接。

启动前可先做一次 Docker 预检查：

```bash
cd ~/project/newton
DISPLAY=:0 docker/run_vr_stack.sh --check-only
```

正常应看到：

```text
[vr-prereqs] ok: preflight passed
```

Docker 启动脚本会把宿主机的 GPU、X11 display、Docker socket、`~/.cloudxr`、
Vosk 模型、CloudXR web cache、`IsaacTeleop` 和本项目目录挂进容器，并在容器内执行：

```bash
scripts/run_newton_vr_prereqs.sh --display :0
```

## 3. Quest 连接

在 Quest 浏览器打开：

```text
https://192.168.8.100:8443/
```

如果证书不通过，先打开：

```text
https://192.168.8.100:48322/
```

接受证书后，再回到 `https://192.168.8.100:8443/`。

进入 XR 页面后，打开语音，然后进入沉浸模式。

## 4. 语音控制

```text
开始    开始遥操
暂停    暂停跟随
继续    恢复跟随
重置    重新锚定
停止    停止跟随并保持
退出    退出遥操
```

也可以在主机上手动发送命令测试：

```bash
scripts/send_teleop_voice_command_once.sh --command engage
scripts/send_teleop_voice_command_once.sh --command clutch
scripts/send_teleop_voice_command_once.sh --command resume
scripts/send_teleop_voice_command_once.sh --command stop
```

## 5. 常用检查

检查 8443 是否启动：

```bash
ss -ltnp | grep 8443
curl -kI https://192.168.8.100:8443/
```

正常应看到 `HTTP/1.1 200 OK`。

查看日志：

```bash
tail -n 80 logs/vr_stack/cloudxr_runtime.log
tail -n 80 logs/vr_stack/cloudxr_web_client.log
tail -n 80 logs/vr_stack/quest_voice_bridge.log
tail -n 80 logs/vr_stack/newton_vr_output.log
```

停止整套流程：在启动终端按 `Ctrl+C`。
