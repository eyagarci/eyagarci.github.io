---
layout: post
title: Zookeeper
date: 2018-12-29
tags: java
---



# Zookeeper

分布式开源框架，解决分布式协调工具

应用场景：dubbo rpc远程调用框架+zookeeper注册中心（命名服务）

发布订阅 对zk节点发生改变，都有事件通知

负载均衡

分布式通知（watcher）生产者到注册中心创建节点信息，消费者通过事件通知进行拿到信息

master选举 主备机 投票选举 ping通就没有宕机 心跳检测机制

Zookeeper分布式锁（全局ID）

使用Zookeeper分布式配置中心

### Zookeeper数据结构

类似于xml的树状结构 节点名称，节点内容 节点名称不可重复

节点类型分为四种：

```
PERSISTENT                    持久化节点
PERSISTENT_SEQUENTIAL        顺序自动编号持久化节点，这种节点会根据当前已存在的节点数自动加 1
EPHEMERAL                     临时节点， 客户端session超时这类节点就会被自动删除
EPHEMERAL_SEQUENTIAL         临时自动编号节点
```

![ss](https://ripperhxy.github.io\images\blog\zookeeper\tree.png)

## Java操作Zookeeper

maven

```xml
<dependency>
			<groupId>org.apache.zookeeper</groupId>
			<artifactId>zookeeper</artifactId>
			<version>3.4.6</version>
		</dependency>
```

java代码

```java
public class Test001 {

	//连接地址
	private static final String ADDRES = "127.0.0.1:2181";
	//session 会话
	private static final int SESSION_OUTTIME = 2000;
	//信号量,阻塞程序执行,用户等待zookeeper连接成功,发送成功信号，
	private static final CountDownLatch countDownLatch = new CountDownLatch(1);

	public static void main(String[] args) throws IOException, InterruptedException, KeeperException {
		ZooKeeper zk = new ZooKeeper(ADDRES, SESSION_OUTTIME, new Watcher() {

			public void process(WatchedEvent event) {
				// 获取事件状态
				KeeperState keeperState = event.getState();
				// 获取事件类型
				EventType eventType = event.getType();
				if (KeeperState.SyncConnected == keeperState) {
					if (EventType.None == eventType) {
						countDownLatch.countDown();
						System.out.println("zk 启动连接...");
					}

				}
			}
		});
		// 进行阻塞
		countDownLatch.await();
		String result = zk.create("/itmayeidu_Lasting", "Lasting".getBytes(), Ids.OPEN_ACL_UNSAFE,
				CreateMode.PERSISTENT);
		System.out.println(result);
		zk.close();
	}
}
```

注意一定要用并发包里进行同步CountDownLatch （类似于计数器）不然会造成还没有启动就对zookeeper进行节点修改

添加节点信息

```

1.	创建持久节点，并且允许任何服务器可以操作
	String result = zk.create("/hxy_Lasting", "Lasting".getBytes(), Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);
	System.out.println("result:" + result);
2.	创建临时节点
	String result = zk.create("/hxy_temp", "temp".getBytes(), Ids.OPEN_ACL_UNSAFE, CreateMode.EPHEMERAL);
	System.out.println("result:" + result)；
	连接释放后临时节点直接也清除了
```

## Watcher

事件通知：节点发生改变的时候	新增、修改、删除的时候都会收到时间通知

那么，zk事件通知有延迟的情况下，怎么处理？

答：**为了降低watch事件延迟带来的信息错误，有如下方法可以参考**：减少修改节点值的用户，一个节点多个观察者是没有问题的，尽量做到一个节点只有一个修改者。可能在实际应用中恰好我们需要这样的机制，多个修改者通过修改一个节点值来通知一个观察者来，观察者针对节点的值做出响应。

| KeeperState            | EventType                 | 触发条件                                                     | 说明                                                         |
| ---------------------- | ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
|                        | None    （-1）            | 客户端与服务端成功建立连接                                   |                                                              |
| SyncConnected    （0） | NodeCreated    （1）      | Watcher监听的对应数据节点被创建                              |                                                              |
|                        | NodeDeleted    （2）      | Watcher监听的对应数据节点被删除                              | 此时客户端和服务器处于连接状态                               |
|                        | NodeDataChanged    （3）  | Watcher监听的对应数据节点的数据内容发生变更                  |                                                              |
|                        | NodeChildChanged    （4） | Wather监听的对应数据节点的子节点列表发生变更                 |                                                              |
| Disconnected    （0）  | None    （-1）            | 客户端与ZooKeeper服务器断开连接                              | 此时客户端和服务器处于断开连接状态                           |
| Expired    （-112）    | Node    （-1）            | 会话超时                                                     | 此时客户端会话失效，通常同时也会受到SessionExpiredException异常 |
| AuthFailed    （4）    | None    （-1）            | 通常有两种情况，1：使用错误的schema进行权限检查 2：SASL权限检查失败 | 通常同时也会收到AuthFailedException异常                      |

Java代码

```java
public class ZkClientWatcher implements Watcher {
	// 集群连接地址
	private static final String CONNECT_ADDRES = "192.168.110.159:2181,192.168.110.160:2181,192.168.110.162:2181";
	// 会话超时时间
	private static final int SESSIONTIME = 2000;
	// 信号量,让zk在连接之前等待,连接成功后才能往下走.
	private static final CountDownLatch countDownLatch = new CountDownLatch(1);
	private static String LOG_MAIN = "【main】 ";
	private ZooKeeper zk;

	public void createConnection(String connectAddres, int sessionTimeOut) {
		try {
			zk = new ZooKeeper(connectAddres, sessionTimeOut, this);
			System.out.println(LOG_MAIN + "zk 开始启动连接服务器....");
			countDownLatch.await();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public boolean createPath(String path, String data) {
		try {
			this.exists(path, true);
			this.zk.create(path, data.getBytes(), Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);
			System.out.println(LOG_MAIN + "节点创建成功, Path:" + path + ",data:" + data);
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		}
		return true;
	}

	/**
	 * 判断指定节点是否存在
	 *
	 * @param path
	 *            节点路径
	 */
	public Stat exists(String path, boolean needWatch) {
		try {
			return this.zk.exists(path, needWatch);
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

	public boolean updateNode(String path,String data) throws KeeperException, InterruptedException {
		exists(path, true);
		this.zk.setData(path, data.getBytes(), -1);
		return false;
	}

	public void process(WatchedEvent watchedEvent) {

		// 获取事件状态
		KeeperState keeperState = watchedEvent.getState();
		// 获取事件类型
		EventType eventType = watchedEvent.getType();
		// zk 路径
		String path = watchedEvent.getPath();
		System.out.println("进入到 process() keeperState:" + keeperState + ", eventType:" + eventType + ", path:" + path);
		// 判断是否建立连接
		if (KeeperState.SyncConnected == keeperState) {
			if (EventType.None == eventType) {
				// 如果建立建立成功,让后程序往下走
				System.out.println(LOG_MAIN + "zk 建立连接成功!");
				countDownLatch.countDown();
			} else if (EventType.NodeCreated == eventType) {
				System.out.println(LOG_MAIN + "事件通知,新增node节点" + path);
			} else if (EventType.NodeDataChanged == eventType) {
				System.out.println(LOG_MAIN + "事件通知,当前node节点" + path + "被修改....");
			}
			else if (EventType.NodeDeleted == eventType) {
				System.out.println(LOG_MAIN + "事件通知,当前node节点" + path + "被删除....");
			}

		}
		System.out.println("--------------------------------------------------------");
	}

	public static void main(String[] args) throws KeeperException, InterruptedException {
		ZkClientWatcher zkClientWatcher = new ZkClientWatcher();
		zkClientWatcher.createConnection(CONNECT_ADDRES, SESSIONTIME);
//		boolean createResult = zkClientWatcher.createPath("/p15", "pa-644064");
		zkClientWatcher.updateNode("/pa2","7894561");
	}

}
```

# Zookeeper实现分布式锁

## 传统方式生成订单号ID

### 业务场景

在分布式情况，生成全局订单号ID

### 生成订单号方案

\1.       使用时间戳

\2.       使用UUID

\3.       推特 (Twitter) 的 Snowflake 算法——用于生成唯一 ID

 订单类

```java
//生成订单类
public class OrderNumGenerator {
    //全局订单id
	public static int count = 0;//要对同一个全局变量进行操作

	public String getNumber() {
		try {
			Thread.sleep(200);
		} catch (Exception e) {
		}
		SimpleDateFormat simpt = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss");
		return simpt.format(new Date()) + "-" + ++count;
	}
}
```

使用多线程情况模拟生成订单号（线程安全问题）

```java
//使用多线程模拟生成订单号
public class OrderService implements Runnable {
	private OrderNumGenerator orderNumGenerator = new OrderNumGenerator();

	public void run() {
		getNumber();
	}

	public void getNumber() {
		String number = orderNumGenerator.getNumber();
		System.out.println(Thread.currentThread().getName() + ",生成订单ID:" + number);
	}

	public static void main(String[] args) {
		System.out.println("####生成唯一订单号###");
		for (int i = 0; i < 100; i++) {
			new Thread(new OrderService()).start();
		}

	}
}
```

同步代码块

```java
//使用多线程模拟生成订单号
public class OrderService implements Runnable {
	private OrderNumGenerator orderNumGenerator = new OrderNumGenerator();

	public void run() {
		getNumber();
	}

	public void getNumber() {
		synchronized (this) {
			String number = orderNumGenerator.getNumber();
			System.out.println(Thread.currentThread().getName() + ",生成订单ID:" + number);
		}
	}

	public static void main(String[] args) {
		System.out.println("####生成唯一订单号###");
		OrderService orderService = new OrderService();
		for (int i = 0; i < 100; i++) {
			new Thread(orderService).start();
		}

	}
}
```

lock锁

```java
public class OrderService implements Runnable {
	private OrderNumGenerator orderNumGenerator = new OrderNumGenerator();
	// 使用lock锁
	private java.util.concurrent.locks.Lock lock = new ReentrantLock();

	public void run() {
		getNumber();
	}

	public void getNumber() {
		try {
			// synchronized (this) {
			lock.lock();
			String number = orderNumGenerator.getNumber();
			System.out.println(Thread.currentThread().getName() + ",生成订单ID:" + number);
			// }

		} catch (Exception e) {

		} finally {
			lock.unlock();
		}
	}

	public static void main(String[] args) {
		System.out.println("####生成唯一订单号###");
		OrderService orderService = new OrderService();
		for (int i = 0; i < 100; i++) {
			new Thread(orderService).start();
		}

	}
}
```

# 分布式场景下生成订单ID





## 业务场景

在分布式情况，生成全局订单号ID

## 产生问题

在分布式(集群)环境下，每台JVM不能实现同步，在分布式场景下使用时间戳生成订单号可能会重复

## 分布式情况下，怎么解决订单号生成不重复

\1.       使用分布式锁

\2.       提前生成好，订单号，存放在redis取。获取订单号，直接从redis中取。

## 使用分布式锁生成订单号技术

1.使用数据库实现分布式锁

缺点:性能差、线程出现异常时，容易出现死锁、释放锁的时候需要自己删除数据、jdbc断开时，不能释放锁，产生死锁的概念

2.使用redis实现分布式锁

缺点:锁的失效时间难控制、容易产生死锁、非阻塞式、不可重入

**3.使用zookeeper实现分布式锁**

**实现相对简单、可靠性强、使用临时节点，失效时间容易控制**

## 什么是分布式锁

分布式锁一般用在分布式系统或者多个应用中，用来控制同一任务是否执行或者任务的执行顺序。在项目中，部署了多个tomcat应用，在执行定时任务时就会遇到同一任务可能执行多次的情况，我们可以借助分布式锁，保证在同一时间只有一个tomcat应用执行了定时任务。

实现原理：在zk创建临时节点，只要谁的服务器能创建节点成功，谁就能获得锁，其他服务没有创建成功，就等待，其他服务器使用事件监听，获取节点通知，如果已经节点被删除，应该获取锁的资源进行循环。

### Zookeeper实现分布式锁原理

使用zookeeper创建临时序列节点来实现分布式锁，适用于顺序执行的程序，大体思路就是创建临时序列节点，找出最小的序列节点，获取分布式锁，程序执行完成之后此序列节点消失，通过watch来监控节点的变化，从剩下的节点的找到最小的序列节点，获取分布式锁，执行相应处理，依次类推……

maven

```xml
	<dependencies>
		<dependency>
			<groupId>com.101tec</groupId>
			<artifactId>zkclient</artifactId>
			<version>0.10</version>
		</dependency>
	</dependencies>
```

### 创建Lock接口

```java
public interface Lock {
    //获取到锁的资源
	public void getLock();
    // 释放锁
	public void unLock();
}
```

### 创建ZookeeperAbstractLock抽象类

```java
//将重复代码写入子类中..
public abstract class ZookeeperAbstractLock implements Lock {
	// zk连接地址
	private static final String CONNECTSTRING = "127.0.0.1:2181";
	// 创建zk连接
	protected ZkClient zkClient = new ZkClient(CONNECTSTRING);
	protected static final String PATH = "/lock";

	public void getLock() {
		if (tryLock()) {
			System.out.println("##获取lock锁的资源####");
		} else {
			// 等待
			waitLock();
			// 重新获取锁资源
			getLock();
		}

	}

	// 获取锁资源
	abstract boolean tryLock();

	// 等待
	abstract void waitLock();

	public void unLock() {
		if (zkClient != null) {
			zkClient.close();
			System.out.println("释放锁资源...");
		}
	}

}
```

### ZookeeperDistrbuteLock类

```java
public class ZookeeperDistrbuteLock extends ZookeeperAbstractLock {
	private CountDownLatch countDownLatch = null;

	@Override
	boolean tryLock() {
		try {
			zkClient.createEphemeral(PATH);
			return true;
		} catch (Exception e) {
//			e.printStackTrace();
			return false;
		}

	}

	@Override
	void waitLock() {
		IZkDataListener izkDataListener = new IZkDataListener() {

			public void handleDataDeleted(String path) throws Exception {
				// 唤醒被等待的线程
				if (countDownLatch != null) {
					countDownLatch.countDown();
				}
			}
			public void handleDataChange(String path, Object data) throws Exception {

			}
		};
		// 注册事件
		zkClient.subscribeDataChanges(PATH, izkDataListener);
		if (zkClient.exists(PATH)) {
			countDownLatch = new CountDownLatch(1);
			try {
				countDownLatch.await();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		// 删除监听
		zkClient.unsubscribeDataChanges(PATH, izkDataListener);
	}

}
```

### 使用Zookeeper锁运行效果

```java
public class OrderService implements Runnable {
	private OrderNumGenerator orderNumGenerator = new OrderNumGenerator();
	// 使用lock锁
	// private java.util.concurrent.locks.Lock lock = new ReentrantLock();
	private Lock lock = new ZookeeperDistrbuteLock();
	public void run() {
		getNumber();
	}
	public void getNumber() {
		try {
			lock.getLock();
			String number = orderNumGenerator.getNumber();
			System.out.println(Thread.currentThread().getName() + ",生成订单ID:" + number);
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			lock.unLock();
		}
	}
	public static void main(String[] args) {
		System.out.println("####生成唯一订单号###");
//		OrderService orderService = new OrderService();
		for (int i = 0; i < 100; i++) {
			new Thread( new OrderService()).start();
		}
	}
}
```

# 使用Zookeeper实现负载均衡原理

使用Zookeeper实现负载均衡原理，服务器端将启动的服务注册到，zk注册中心上，采用临时节点。客户端从zk节点上获取最新服务节点信息，本地使用负载均衡算法，随机分配服务器。

### Maven依赖

```xml
	<dependencies>
		<dependency>
			<groupId>com.101tec</groupId>
			<artifactId>zkclient</artifactId>
			<version>0.8</version>
		</dependency>
	</dependencies>
```

### ZkServerScoekt服务

```java
//##ServerScoekt服务端
public class ZkServerScoekt implements Runnable {
	private int port = 18080;

	public static void main(String[] args) throws IOException {
		int port = 18080;
		ZkServerScoekt server = new ZkServerScoekt(port);
		Thread thread = new Thread(server);
		thread.start();
	}

	public ZkServerScoekt(int port) {
		this.port = port;
	}

	public void run() {
		ServerSocket serverSocket = null;
		try {
			serverSocket = new ServerSocket(port);
			System.out.println("Server start port:" + port);
			Socket socket = null;
			while (true) {
				socket = serverSocket.accept();
				new Thread(new ServerHandler(socket)).start();
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				if (serverSocket != null) {
					serverSocket.close();
				}
			} catch (Exception e2) {

			}
		}
	}

}
```

### zkClient

```java
public class ZkServerClient {
	public static List<String> listServer = new ArrayList<String>();

	public static void main(String[] args) {
		initServer();
		ZkServerClient 	client= new ZkServerClient();
		BufferedReader console = new BufferedReader(new InputStreamReader(System.in));
		while (true) {
			String name;
			try {
				name = console.readLine();
				if ("exit".equals(name)) {
					System.exit(0);
				}
				client.send(name);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	// 注册所有server
	public static void initServer() {
		listServer.clear();
		listServer.add("127.0.0.1:18080");
	}

	// 获取当前server信息
	public static String getServer() {
		return listServer.get(0);
	}

	public void send(String name) {

		String server = ZkServerClient.getServer();
		String[] cfg = server.split(":");

		Socket socket = null;
		BufferedReader in = null;
		PrintWriter out = null;
		try {
			socket = new Socket(cfg[0], Integer.parseInt(cfg[1]));
			in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
			out = new PrintWriter(socket.getOutputStream(), true);

			out.println(name);
			while (true) {
				String resp = in.readLine();
				if (resp == null)
					break;
				else if (resp.length() > 0) {
					System.out.println("Receive : " + resp);
					break;
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (out != null) {
				out.close();
			}
			if (in != null) {
				try {
					in.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			if (socket != null) {
				try {
					socket.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}
}

```

负载均衡轮询机制使用取模算法：调用次数除以服务器集群数量=服务位置（1%2=1|2%2=0|3%2=1|4%2=0）

### 改造ZkServerScoekt

```java
public class ZkServerScoekt implements Runnable {
	private static int port = 18081;

	public static void main(String[] args) throws IOException {
		ZkServerScoekt server = new ZkServerScoekt(port);
		Thread thread = new Thread(server);
		thread.start();
	}

	public ZkServerScoekt(int port) {
		this.port = port;
	}

	public void regServer() {
		// 向ZooKeeper注册当前服务器
		ZkClient client = new ZkClient("127.0.0.1:2181", 60000, 1000);
		String path = "/test/server" + port;
		if (client.exists(path))
			client.delete(path);
		client.createEphemeral(path, "127.0.0.1:" + port);
	}

	public void run() {
		ServerSocket serverSocket = null;
		try {
			serverSocket = new ServerSocket(port);
			regServer();
			System.out.println("Server start port:" + port);
			Socket socket = null;
			while (true) {
				socket = serverSocket.accept();
				new Thread(new ServerHandler(socket)).start();
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				if (serverSocket != null) {
					serverSocket.close();
				}
			} catch (Exception e2) {

			}
		}
	}

}
```

### 改造ZKClient

```java
public class ZkServerClient {
	public static List<String> listServer = new ArrayList<String>();

	public static void main(String[] args) {
		initServer();
		ZkServerClient client = new ZkServerClient();
		BufferedReader console = new BufferedReader(new InputStreamReader(System.in));
		while (true) {
			String name;
			try {
				name = console.readLine();
				if ("exit".equals(name)) {
					System.exit(0);
				}
				client.send(name);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	// 注册所有server
	public static void initServer() {
		final String path = "/test";
		final ZkClient zkClient = new ZkClient("127.0.0.1:2181", 60000, 1000);
		List<String> children = zkClient.getChildren(path);
		listServer.clear();
		for (String p : children) {
			listServer.add((String) zkClient.readData(path + "/" + p));
		}
		// 订阅节点变化事件
		zkClient.subscribeChildChanges("/test", new IZkChildListener() {

			public void handleChildChange(String parentPath, List<String> currentChilds) throws Exception {
				listServer.clear();
				for (String p : currentChilds) {
					listServer.add((String) zkClient.readData(path + "/" + p));
				}
				System.out.println("####handleChildChange()####listServer:" + listServer.toString());
			}
		});
	}

	// 请求次数
	private static int count = 1;
	// 服务数量
	private static int serverCount=2;

	// 获取当前server信息
	public static String getServer() {
		String serverName = listServer.get(count%serverCount);
		++count;
		return serverName;
	}

	public void send(String name) {

		String server = ZkServerClient.getServer();
		String[] cfg = server.split(":");

		Socket socket = null;
		BufferedReader in = null;
		PrintWriter out = null;
		try {
			socket = new Socket(cfg[0], Integer.parseInt(cfg[1]));
			in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
			out = new PrintWriter(socket.getOutputStream(), true);

			out.println(name);
			while (true) {
				String resp = in.readLine();
				if (resp == null)
					break;
				else if (resp.length() > 0) {
					System.out.println("Receive : " + resp);
					break;
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (out != null) {
				out.close();
			}
			if (in != null) {
				try {
					in.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			if (socket != null) {
				try {
					socket.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}
}
```

# 使用Zookeeper实现选举策略

有一个向外提供的服务，服务必须7*24小时提供服务，不能有单点故障。所以采用集群的方式，采用master、slave的结构。一台主机多台备机。主机向外提供服务，备机负责监听主机的状态，一旦主机宕机，备机要迅速接代主机继续向外提供服务。从备机选择一台作为主机，就是master选举。

## 原理分析

得到事件通知后，备机服务器会尝试创建master节点，谁创建成功了，就是master，向外提供。其他两台就是slave。

所有slave必须关注master的删除事件（临时节点，如果服务器宕机了，Zookeeper会自动把master节点删除）。如果master宕机了，会进行新一轮的master选举。
