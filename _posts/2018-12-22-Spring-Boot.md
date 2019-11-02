---
layout: post
title: Spring Boot
date: 2018-12-22
tags: java
---



# SpringBoot

## SpringBoot（主流--安全）

1、敏捷开发：整合任何框架，配置基本都封装好， 只需依赖主键

2、无需Tomcat

3、减少xml配置 没有（xml），换成了配置文件

4、SpringBoot和SpringCloud联合开发（微服务---->SpringCloud【http接口+restful】，基于SpringBoot web组件封装Springmvc）

5、注解--记住

## 创建SpringBoot工程

### 1、建立类型为Jar的Maven工程

### 2、xml配置

```xml
	<parent>
		<groupId>org.springframework.boot</groupId>
		<artifactId>spring-boot-starter-parent</artifactId>
		<version>1.3.3.RELEASE</version>
	</parent>
	<dependencies>
	  <!—SpringBoot web 组件 -->
		<dependency>
			<groupId>org.springframework.boot</groupId>
			<artifactId>spring-boot-starter-web</artifactId>
		</dependency>
	</dependencies>

```

#### spring-boot-starter-parent作用

在pom.xml中引入spring-boot-start-parent,spring官方的解释叫什么stater poms,它可以提供dependency management,也就是说依赖管理，引入以后在申明其它dependency的时候就不需要version了，可以点进去看。

#### spring-boot-starter-web作用

springweb核心组件

#### spring-boot-maven-plugin作用

 如果我们要直接Main启动spring，那么以下plugin必须要添加，否则是无法启动的。如果使用maven 的spring-boot:run的话是不需要此配置的。

---

### 3、创建我们的第一个接口

第一种启动方式

```java
//标识该接口全部返回json格式
@RestController
@EnableAutoConfiguration
public class HelloController {
	@RequestMapping("/hello")
	public String index() {
		return "Hello World";
	}
public static void main(String[] args) {
		SpringApplication.run(HelloController.class, args);
	}
}

```

@EnableAutoConfiguration注解:作用在于让 Spring Boot   根据应用所声明的依赖来对 Spring 框架进行自动配置
 这个注解告诉Spring Boot根据添加的jar依赖猜测你想如何配置Spring。由于spring-boot-starter-web添加了Tomcat和Spring MVC，所以auto-configuration将假定你正在开发一个web应用并相应地对Spring进行设置。   **表示注入spring容器、创建tomcat、spring加载**

---

第二种启动方式

```java
@ComponentScan(basePackages = "com.hxy.controller")//扫描包
@EnableAutoConfiguration
public class App {
	public static void main(String[] args) {
		SpringApplication.run(App.class, args);
	}
}

```

只能用有一个main函数。所以当多个类需要接口的时候可以创建一个启动类来专门启动springboot。

## web开发

### 静态资源的访问

在我们开发Web应用的时候，需要引用大量的js、css、图片等静态资源。

默认配置

Spring Boot默认提供静态资源目录位置需置于classpath下，目录名需符合如下规则：

/static

/public

/resources          

/META-INF/resources

举例：我们可以在src/main/resources/目录下创建static，在该位置放置一个图片文件D.jpg。启动程序后，尝试访问http://localhost:8080/D.jpg。如能显示图片，配置成功。

### 全局捕获异常

#### @ExceptionHandler

表示拦截异常

#### @ControllerAdvice

是 controller 的一个辅助类，最常用的就是作为全局异常处理的切面类 可以指定扫描范围

约定了几种可行的返回值，如果是直接返回 model 类的话，需要使用 @ResponseBody 进行 json 转换

返回 String，表示跳到某个 view

返回 modelAndView

返回 model + @ResponseBody

## 渲染Web页面

SpringBoot优先模板引擎，不建议使用jsp，那么什么是模板引擎呢？

就是把一个动态jsp页面伪造成html格式，能提高搜索引擎的搜索，能做动态html实现。

freemarker 、Thymeleaf 、Velocity、Mustache动态页面静态化

### 举例

#### 使用Freemarker模板引擎渲染web视图

1、配置pom.xml

```xml
<!-- 引入freeMarker的依赖包. -->
<dependency>
<groupId>org.springframework.boot</groupId>
<artifactId>spring-boot-starter-freemarker</artifactId>
</dependency>

```

2、前段代码

在src/main/resources/创建一个templates文件夹,后缀为*.ftl

```ft
hello SpringBoot
	  ${name}
```

3、后台代码

```java
	//上面直接controller就可以  不需要restcontroller
	@RequestMapping("/index")
	public String index(Map<String, Object> map) {
	    map.put("name","hxy");
		return "index";
	}
```

#### freemarker配置文件

新建application.properties文件

```properties
########################################################
###FREEMARKER (FreeMarkerAutoConfiguration)
########################################################
spring.freemarker.allow-request-override=false
spring.freemarker.cache=true
spring.freemarker.check-template-location=true
spring.freemarker.charset=UTF-8
spring.freemarker.content-type=text/html
spring.freemarker.expose-request-attributes=false
spring.freemarker.expose-session-attributes=false
spring.freemarker.expose-spring-macro-helpers=false
#spring.freemarker.prefix=
#spring.freemarker.request-context-attribute=
#spring.freemarker.settings.*=
spring.freemarker.suffix=.ftl
spring.freemarker.template-loader-path=classpath:/templates/
#comma-separated list
#spring.freemarker.view-names= # whitelist of view names that can be resolved

```

#### 注意

springboot渲染jsp就不赘述了，外面公司都是渲染html。

### springboot整合jdbcTemplete

新增pom.xml

```xml
<dependencies>
		<dependency>
			<groupId>org.springframework.boot</groupId>
			<artifactId>spring-boot-starter-jdbc</artifactId>
		</dependency>
		<dependency>
			<groupId>mysql</groupId>
			<artifactId>mysql-connector-java</artifactId>
		</dependency>
</dependencies>
```

新增applianction.properties

```properties
spring.datasource.url=jdbc:mysql://localhost:3306/test
spring.datasource.username=root
spring.datasource.password=root
spring.datasource.driver-class-name=com.mysql.jdbc.Driver

```

代码

```java
@Service
public class UserServiceImpl implements UserService {
	@Autowired
	private JdbcTemplate jdbcTemplate;
	public void createUser(String name, Integer age) {
		System.out.println("ssss");
		jdbcTemplate.update("insert into users values(null,?,?);", name, age);
	}
}

```

记住要扫包不仅仅是controller 还有service

---

### springboot整合mybaties

新增pom.xml

```xml
	<parent>
		<groupId>org.springframework.boot</groupId>
		<artifactId>spring-boot-starter-parent</artifactId>
		<version>1.3.2.RELEASE</version>
		<relativePath /> <!-- lookup parent from repository -->
	</parent>
	<dependencies>
		<dependency>
			<groupId>org.springframework.boot</groupId>
			<artifactId>spring-boot-starter</artifactId>
		</dependency>
		<dependency>
			<groupId>org.springframework.boot</groupId>
			<artifactId>spring-boot-starter-test</artifactId>
			<scope>test</scope>
		</dependency>
		<dependency>
			<groupId>org.mybatis.spring.boot</groupId>
			<artifactId>mybatis-spring-boot-starter</artifactId>
			<version>1.1.1</version>
		</dependency>
		<dependency>
			<groupId>mysql</groupId>
			<artifactId>mysql-connector-java</artifactId>
			<version>5.1.21</version>
		</dependency>
		<dependency>
			<groupId>org.springframework.boot</groupId>
			<artifactId>spring-boot-starter-web</artifactId>
		</dependency>
	</dependencies>

```

新增application.properties

```properties
spring.datasource.url=jdbc:mysql://localhost:3306/test
spring.datasource.username=root
spring.datasource.password=root
spring.datasource.driver-class-name=com.mysql.jdbc.Driver
```

mapper

```java
public interface UserMapper {
	@Select("SELECT * FROM USERS WHERE NAME = #{name}")
	User findByName(@Param("name") String name);
	@Insert("INSERT INTO USERS(NAME, AGE) VALUES(#{name}, #{age})")
	int insert(@Param("name") String name, @Param("age") Integer age);
}

```

启动新加注解

```java
@ComponentScan(basePackages = "com.hxy")
@MapperScan(basePackages = "com.hxy.mapper")
@SpringBootApplication
public class App {
	public static void main(String[] args) {
		SpringApplication.run(App.class, args);
	}
}
```

---

### springboot整合jpa

新增pom.xml

```xml
		<dependency>
			<groupId>org.springframework.boot</groupId>
			<artifactId>spring-boot-starter-data-jpa</artifactId>
		</dependency>
		<dependency>
			<groupId>mysql</groupId>
			<artifactId>mysql-connector-java</artifactId>
			<version>5.1.21</version>
		</dependency>

```

springboot整合jdbc框架，不会产生冲突

创建实体类

```java
@Entity(name = "users")//表名称
public class User {
	@Id
	@GeneratedValue
	private Integer id;
	@Column
	private String name;
	@Column
	private Integer age;
      // ..get/set方法
}

```

新增Dao层

```java
public interface UserDao extends JpaRepository<User, Integer> {
}
```

集成了这个类之后，基本的操作数据库的api都有了

创建controller

```java
@RestController
public class IndexController {
	@Autowired
	private UserDao userDao;
	@RequestMapping("/index")
	public String index(Integer id) {
		User findUser = userDao.findOne(id);
		System.out.println(findUser.getName());
		return "success";
	}
}
```

启动

```java
@ComponentScan(basePackages = { "com.hxy" })
@EnableJpaRepositories(basePackages = "com.hxy.dao")//jpa扫包
@EnableAutoConfiguration
@EntityScan(basePackages = "com.hxy.entity")//扫描实类
public class App {
	public static void main(String[] args) {
		SpringApplication.run(App.class, args);
	}
}

```

## springboot高级

### 注解：@SpringBootApplication

启动项目，整合常用注解，扫描包，将App.class启动类放在最外面

### Springboot整合多个数据源

在实际项目中，怎么搭建多数据源，区分数据源

1、分包结构

例如：两个数据源 db1、db2

com.hxy.db1-------------------访问db1

com.hxy.db2-------------------访问db2

分布式事物解决方案 jta+automatic 传统项目解决

2、使用注解方式

```java
@DataSourcedb1--自定义注解
public void db1（）{}
@DataSourcedb1
public void db1（）{}
```

常用分包结构

配置文件中新增两个数据源

```properties
spring.datasource.test1.driverClassName = com.mysql.jdbc.Driver
spring.datasource.test1.url = jdbc:mysql://localhost:3306/test01?useUnicode=true&characterEncoding=utf-8
spring.datasource.test1.username = root
spring.datasource.test1.password = root

spring.datasource.test2.driverClassName = com.mysql.jdbc.Driver
spring.datasource.test2.url = jdbc:mysql://localhost:3306/test02?useUnicode=true&characterEncoding=utf-8
spring.datasource.test2.username = root
spring.datasource.test2.password = root

```

数据源配置代码

```java
@Configuration // 注册到springboot容器中
@MapperScan(basePackages = "com.hxy.user1", sqlSessionFactoryRef = "test1SqlSessionFactory")
public class DataSource1Config {


	@Bean(name = "test1DataSource")
	@Primary//springboot启动的时候首先选择的数据源必须要配置，不然就会找不到到底用哪个数据源
	@ConfigurationProperties(prefix = "spring.datasource.test1")//配置文件的名字
	public DataSource testDataSource() {
		return DataSourceBuilder.create().build();
	}

	@Bean(name = "test1SqlSessionFactory")
	@Primary
	public SqlSessionFactory testSqlSessionFactory(@Qualifier("test1DataSource") DataSource dataSource)
			throws Exception {
		SqlSessionFactoryBean bean = new SqlSessionFactoryBean();
		bean.setDataSource(dataSource);
//		bean.setMapperLocations(
//				new PathMatchingResourcePatternResolver().getResources("classpath:mybatis/mapper/test1/*.xml"));
		return bean.getObject();
	}


	@Bean(name = "test1TransactionManager")
	@Primary
	public DataSourceTransactionManager testTransactionManager(@Qualifier("test1DataSource") DataSource dataSource) {
		return new DataSourceTransactionManager(dataSource);
	}

	@Bean(name = "test1SqlSessionTemplate")
	public SqlSessionTemplate testSqlSessionTemplate(
			@Qualifier("test1SqlSessionFactory") SqlSessionFactory sqlSessionFactory) throws Exception {
		return new SqlSessionTemplate(sqlSessionFactory);
	}
}
```

---

### 事务管理

Spring事务分类：编程事务，声明式事务xml+注解；

springboot默认集成事物,只主要在方法上加上@Transactional即可

#### 分布式事务

一个方法里面有两个调不同数据源的方法，一个通过mapper调用，另一个通过service调用，两个都加上了事务注解，添加int=1/0后，都实现了事务回滚，但是都调用mapper的时候，不是主要类下面的mapper还是添加成功了，怎么操作才能实现都管理到事务？

更新pom.xml

```xml
<dependency>
	<groupId>org.springframework.boot</groupId>
	<artifactId>spring-boot-starter-jta-atomikos</artifactId>
</dependency>

```

#### 新增配置文件信息

```properties
# Mysql 1
mysql.datasource.test.url = jdbc:mysql://localhost:3306/test01?useUnicode=true&characterEncoding=utf-8
mysql.datasource.test.username = root
mysql.datasource.test.password = root

mysql.datasource.test.minPoolSize = 3
mysql.datasource.test.maxPoolSize = 25
mysql.datasource.test.maxLifetime = 20000
mysql.datasource.test.borrowConnectionTimeout = 30
mysql.datasource.test.loginTimeout = 30
mysql.datasource.test.maintenanceInterval = 60
mysql.datasource.test.maxIdleTime = 60
mysql.datasource.test.testQuery = select 1


# Mysql 2
mysql.datasource.test2.url =jdbc:mysql://localhost:3306/test02?useUnicode=true&characterEncoding=utf-8
mysql.datasource.test2.username =root
mysql.datasource.test2.password =root

mysql.datasource.test2.minPoolSize = 3
mysql.datasource.test2.maxPoolSize = 25
mysql.datasource.test2.maxLifetime = 20000
mysql.datasource.test2.borrowConnectionTimeout = 30
mysql.datasource.test2.loginTimeout = 30
mysql.datasource.test2.maintenanceInterval = 60
mysql.datasource.test2.maxIdleTime = 60
mysql.datasource.test2.testQuery = select 1


```

#### 读取配置文件

```
package com.hxy.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "mysql.datasource.test")//读取对应数据库
public class DBConfig1 {

	private String url;
	private String username;
	private String password;
	private int minPoolSize;
	private int maxPoolSize;
	private int maxLifetime;
	private int borrowConnectionTimeout;
	private int loginTimeout;
	private int maintenanceInterval;
	private int maxIdleTime;
	private String testQuery;
}
```

```
package com.hxy.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "mysql.datasource.test1")//读取对应数据库
public class DBConfig2 {

	private String url;
	private String username;
	private String password;
	private int minPoolSize;
	private int maxPoolSize;
	private int maxLifetime;
	private int borrowConnectionTimeout;
	private int loginTimeout;
	private int maintenanceInterval;
	private int maxIdleTime;
	private String testQuery;
}

```

创建多数据源

```java
@Configuration
// basePackages 最好分开配置 如果放在同一个文件夹可能会报错
@MapperScan(basePackages = "com.hxy.test01", sqlSessionTemplateRef = "testSqlSessionTemplate")
public class TestMyBatisConfig1 {

	// 配置数据源
	@Primary
	@Bean(name = "testDataSource")
	public DataSource testDataSource(DBConfig1 testConfig) throws SQLException {
		MysqlXADataSource mysqlXaDataSource = new MysqlXADataSource();
		mysqlXaDataSource.setUrl(testConfig.getUrl());
		mysqlXaDataSource.setPinGlobalTxToPhysicalConnection(true);
		mysqlXaDataSource.setPassword(testConfig.getPassword());
		mysqlXaDataSource.setUser(testConfig.getUsername());
		mysqlXaDataSource.setPinGlobalTxToPhysicalConnection(true);

		AtomikosDataSourceBean xaDataSource = new AtomikosDataSourceBean();
		xaDataSource.setXaDataSource(mysqlXaDataSource);
		xaDataSource.setUniqueResourceName("testDataSource");

		xaDataSource.setMinPoolSize(testConfig.getMinPoolSize());
		xaDataSource.setMaxPoolSize(testConfig.getMaxPoolSize());
		xaDataSource.setMaxLifetime(testConfig.getMaxLifetime());
		xaDataSource.setBorrowConnectionTimeout(testConfig.getBorrowConnectionTimeout());
		xaDataSource.setLoginTimeout(testConfig.getLoginTimeout());
		xaDataSource.setMaintenanceInterval(testConfig.getMaintenanceInterval());
		xaDataSource.setMaxIdleTime(testConfig.getMaxIdleTime());
		xaDataSource.setTestQuery(testConfig.getTestQuery());
		return xaDataSource;
	}

	@Bean(name = "testSqlSessionFactory")
	public SqlSessionFactory testSqlSessionFactory(@Qualifier("testDataSource") DataSource dataSource)
			throws Exception {
		SqlSessionFactoryBean bean = new SqlSessionFactoryBean();
		bean.setDataSource(dataSource);
		return bean.getObject();
	}

	@Bean(name = "testSqlSessionTemplate")
	public SqlSessionTemplate testSqlSessionTemplate(
			@Qualifier("testSqlSessionFactory") SqlSessionFactory sqlSessionFactory) throws Exception {
		return new SqlSessionTemplate(sqlSessionFactory);
	}
}
///////////////////////////////////////////////////////////////////////////////////////

@Configuration
// basePackages 最好分开配置 如果放在同一个文件夹可能会报错
@MapperScan(basePackages = "com.hxy.test02", sqlSessionTemplateRef = "test2SqlSessionTemplate")
public class TestMyBatisConfig2 {

	// 配置数据源
	@Bean(name = "test2DataSource")
	public DataSource testDataSource(DBConfig1 testConfig) throws SQLException {
		MysqlXADataSource mysqlXaDataSource = new MysqlXADataSource();
		mysqlXaDataSource.setUrl(testConfig.getUrl());
		mysqlXaDataSource.setPinGlobalTxToPhysicalConnection(true);
		mysqlXaDataSource.setPassword(testConfig.getPassword());
		mysqlXaDataSource.setUser(testConfig.getUsername());
		mysqlXaDataSource.setPinGlobalTxToPhysicalConnection(true);

		AtomikosDataSourceBean xaDataSource = new AtomikosDataSourceBean();
		xaDataSource.setXaDataSource(mysqlXaDataSource);
		xaDataSource.setUniqueResourceName("test2DataSource");

		xaDataSource.setMinPoolSize(testConfig.getMinPoolSize());
		xaDataSource.setMaxPoolSize(testConfig.getMaxPoolSize());
		xaDataSource.setMaxLifetime(testConfig.getMaxLifetime());
		xaDataSource.setBorrowConnectionTimeout(testConfig.getBorrowConnectionTimeout());
		xaDataSource.setLoginTimeout(testConfig.getLoginTimeout());
		xaDataSource.setMaintenanceInterval(testConfig.getMaintenanceInterval());
		xaDataSource.setMaxIdleTime(testConfig.getMaxIdleTime());
		xaDataSource.setTestQuery(testConfig.getTestQuery());
		return xaDataSource;
	}

	@Bean(name = "test2SqlSessionFactory")
	public SqlSessionFactory testSqlSessionFactory(@Qualifier("test2DataSource") DataSource dataSource)
			throws Exception {
		SqlSessionFactoryBean bean = new SqlSessionFactoryBean();
		bean.setDataSource(dataSource);
		return bean.getObject();
	}

	@Bean(name = "test2SqlSessionTemplate")
	public SqlSessionTemplate testSqlSessionTemplate(
			@Qualifier("test2SqlSessionFactory") SqlSessionFactory sqlSessionFactory) throws Exception {
		return new SqlSessionTemplate(sqlSessionFactory);
	}
}

```

启动项目

必须要将配置信息的代码扫描

```java
@EnableConfigurationProperties(value = { DBConfig1.class, DBConfig2.class })
```

### springboot整合日志

采用lombok的时候再写

### 使用AOP统一处理Web请求日志

aop依赖

```xml
<dependency>
			<groupId>org.springframework.boot</groupId>
			<artifactId>spring-boot-starter-aop</artifactId>
</dependency>

```

代码

```java
@Aspect
@Component
public class WebLogAspect {
	private Logger logger = LoggerFactory.getLogger(getClass());
	@Pointcut("execution(public * com.hxy.controller..*.*(..))")
	public void webLog() {
	}
	@Before("webLog()")
	public void doBefore(JoinPoint joinPoint) throws Throwable {
		// 接收到请求，记录请求内容
		ServletRequestAttributes attributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
		HttpServletRequest request = attributes.getRequest();
		// 记录下请求内容
		logger.info("URL : " + request.getRequestURL().toString());
		logger.info("HTTP_METHOD : " + request.getMethod());
		logger.info("IP : " + request.getRemoteAddr());
		Enumeration<String> enu = request.getParameterNames();
		while (enu.hasMoreElements()) {
			String name = (String) enu.nextElement();
			logger.info("name:{},value:{}", name, request.getParameter(name));
		}
	}
	@AfterReturning(returning = "ret", pointcut = "webLog()")
	public void doAfterReturning(Object ret) throws Throwable {
		// 处理完请求，返回内容
		logger.info("RESPONSE : " + ret);
	}
}

```

---

### 使用@Async实现异步调用

启动加上@EnableAsync ,需要执行异步方法上加入  @Async

### 自定义参数

配置文件值

```properties
name=hxy.com
```

代码

```java
	@Value("${name}")
	private String name;
	@ResponseBody
	@RequestMapping("/getValue")
	public String getValue() {
		return name；
	}
```

### 多环境开发

spring.profiles.active=pre

application-dev.properties：开发环境

application-test.properties：测试环境

application-prod.properties：生产环境

### 修改端口号

server.port=8888

server.context-path=/hxy

### SpringBoot yml 使用----主流

```ym
server:
  port:  8090
  context-path: /hxy
```

冒号后面加空格 有提示 主流就用这个替代properties

### springboot打包

选中项目点击右键show in Terminal

使用mvn package 打包

cmd

使用java –jar 包名

如果报错没有主清单,在pom文件中新增

```xml
	<build>
		<plugins>
			<plugin>
				<groupId>org.apache.maven.plugins</groupId>
				<artifactId>maven-compiler-plugin</artifactId>
				<configuration>
					<source>1.8</source>
					<target>1.8</target>
				</configuration>
			</plugin>
			<plugin>
				<groupId>org.springframework.boot</groupId>
				<artifactId>spring-boot-maven-plugin</artifactId>
				<configuration>
					<maimClass>com.hxy.app.App</maimClass>
				</configuration>
				<executions>
					<execution>
						<goals>
							<goal>repackage</goal>
						</goals>
					</execution>
				</executions>

			</plugin>
		</plugins>
	</build>


```
