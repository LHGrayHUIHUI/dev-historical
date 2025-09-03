# Story 1.2: 数据获取服务开发

## 基本信息
- **Story ID**: 1.2
- **Epic**: Epic 1 - 微服务基础架构和数据采集
- **标题**: 数据获取服务开发
- **优先级**: 高
- **状态**: ✅ 已完成
- **预估工期**: 4-5天
- **完成时间**: 2025-09-03

## 用户故事
**作为** 数据获取专员  
**我希望** 有一个独立的数据获取服务  
**以便** 从多个平台自动获取和管理内容数据

## 需求描述
开发独立的数据获取微服务，支持多平台内容爬取、代理管理、反封禁策略，实现智能化的数据获取和内容管理功能。

## 技术实现

### 核心技术栈
- **后端框架**: NestJS 10+ (Node.js) - 企业级框架，支持依赖注入和模块化
- **数据库**: PostgreSQL 15+ - ACID事务支持，UUID主键，JSON字段支持
- **缓存**: Redis 7+ - 会话存储，令牌黑名单，限流计数
- **认证**: JWT + Refresh Token - RS256算法，短期访问令牌(15分钟) + 长期刷新令牌(7天)
- **加密**: bcrypt (密码哈希), crypto (随机数生成), argon2 (高安全性哈希)
- **验证**: class-validator, class-transformer - DTO验证和数据转换
- **ORM**: TypeORM 0.3+ - 数据库抽象层，迁移管理，查询构建器
- **API文档**: Swagger/OpenAPI 3.0 - 自动生成API文档，支持认证测试
- **监控**: Prometheus + Grafana - 认证指标监控，性能分析
- **日志**: Winston + ELK Stack - 结构化日志，安全事件追踪
- **限流**: Redis + 滑动窗口算法 - 防止暴力破解攻击
- **消息队列**: Redis Pub/Sub - 用户状态变更通知

### 数据模型设计

#### 用户表 (users)
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    avatar_url VARCHAR(255),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended', 'pending')),
    email_verified BOOLEAN DEFAULT FALSE,
    phone VARCHAR(20),
    phone_verified BOOLEAN DEFAULT FALSE,
    two_factor_enabled BOOLEAN DEFAULT FALSE,
    two_factor_secret VARCHAR(32),
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    last_login_at TIMESTAMP,
    last_login_ip INET,
    password_changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 索引优化
    CONSTRAINT users_username_length CHECK (length(username) >= 3),
    CONSTRAINT users_email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

-- 创建索引
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_last_login ON users(last_login_at);
CREATE INDEX idx_users_metadata ON users USING GIN(metadata);
```

#### 角色表 (roles)
```sql
CREATE TABLE roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(50) UNIQUE NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    description TEXT,
    permissions JSONB NOT NULL DEFAULT '[]',
    is_system_role BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT roles_name_format CHECK (name ~* '^[a-z][a-z0-9_]*$')
);

-- 创建索引
CREATE INDEX idx_roles_name ON roles(name);
CREATE INDEX idx_roles_active ON roles(is_active);
CREATE INDEX idx_roles_permissions ON roles USING GIN(permissions);

-- 插入默认角色
INSERT INTO roles (name, display_name, description, permissions, is_system_role) VALUES
('super_admin', '超级管理员', '系统超级管理员，拥有所有权限', '["*"]', TRUE),
('admin', '管理员', '系统管理员，拥有大部分管理权限', '["users:*", "content:*", "system:read"]', TRUE),
('editor', '编辑员', '内容编辑员，可以管理内容', '["content:read", "content:write", "content:publish"]', TRUE),
('user', '普通用户', '普通用户，基础访问权限', '["content:read", "profile:write"]', TRUE),
('guest', '访客', '访客用户，只读权限', '["content:read"]', TRUE);
```

#### 用户角色关联表 (user_roles)
```sql
CREATE TABLE user_roles (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID REFERENCES roles(id) ON DELETE CASCADE,
    assigned_by UUID REFERENCES users(id),
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    PRIMARY KEY (user_id, role_id)
);

-- 创建索引
CREATE INDEX idx_user_roles_user_id ON user_roles(user_id);
CREATE INDEX idx_user_roles_role_id ON user_roles(role_id);
CREATE INDEX idx_user_roles_active ON user_roles(is_active);
CREATE INDEX idx_user_roles_expires ON user_roles(expires_at);
```

#### 用户会话表 (user_sessions)
```sql
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token_hash VARCHAR(255),
    device_info JSONB,
    ip_address INET,
    user_agent TEXT,
    location JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_sessions_token ON user_sessions(session_token);
CREATE INDEX idx_sessions_active ON user_sessions(is_active);
CREATE INDEX idx_sessions_expires ON user_sessions(expires_at);
```

#### 安全日志表 (security_logs)
```sql
CREATE TABLE security_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB,
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN,
    risk_score INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_security_logs_user_id ON security_logs(user_id);
CREATE INDEX idx_security_logs_event_type ON security_logs(event_type);
CREATE INDEX idx_security_logs_created_at ON security_logs(created_at);
CREATE INDEX idx_security_logs_risk_score ON security_logs(risk_score);
```

### 服务架构

#### 认证服务结构 (基于企业级安全架构)

```typescript
// src/auth/auth.service.ts
import { Injectable, UnauthorizedException, BadRequestException, Logger } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { JwtService } from '@nestjs/jwt';
import { ConfigService } from '@nestjs/config';
import * as bcrypt from 'bcrypt';
import * as crypto from 'crypto';
import { User, UserSession, SecurityLog } from './entities';
import { RedisService } from '../redis/redis.service';
import { RateLimiterService } from '../rate-limiter/rate-limiter.service';
import { SecurityService } from '../security/security.service';
import { NotificationService } from '../notification/notification.service';

@Injectable()
export class AuthService {
  private readonly logger = new Logger(AuthService.name);
  private readonly maxFailedAttempts = 5;
  private readonly lockoutDuration = 15 * 60 * 1000; // 15分钟

  constructor(
    @InjectRepository(User)
    private userRepository: Repository<User>,
    @InjectRepository(UserSession)
    private sessionRepository: Repository<UserSession>,
    @InjectRepository(SecurityLog)
    private securityLogRepository: Repository<SecurityLog>,
    private jwtService: JwtService,
    private redisService: RedisService,
    private rateLimiterService: RateLimiterService,
    private securityService: SecurityService,
    private notificationService: NotificationService,
    private configService: ConfigService,
  ) {}

  /**
   * 用户登录认证 - 包含完整的安全检查
   * @param loginDto 登录信息
   * @param clientInfo 客户端信息
   * @returns 认证结果和令牌
   */
  async login(loginDto: LoginDto, clientInfo: ClientInfo): Promise<AuthResult> {
    const { username, password, captcha, twoFactorCode } = loginDto;
    const { ipAddress, userAgent, deviceFingerprint } = clientInfo;

    // 1. 速率限制检查
    await this.rateLimiterService.checkLimit(`login:${ipAddress}`, 5, 300); // 5次/5分钟

    // 2. 验证码检查 (如果需要)
    if (captcha) {
      await this.securityService.verifyCaptcha(captcha);
    }

    // 3. 查找用户
    const user = await this.findUserByUsernameOrEmail(username);
    if (!user) {
      await this.logSecurityEvent('LOGIN_FAILED', { username, reason: 'user_not_found' }, clientInfo);
      throw new UnauthorizedException('用户名或密码错误');
    }

    // 4. 检查账户状态
    await this.checkAccountStatus(user);

    // 5. 验证密码
    const isPasswordValid = await bcrypt.compare(password, user.passwordHash);
    if (!isPasswordValid) {
      await this.handleFailedLogin(user, clientInfo);
      throw new UnauthorizedException('用户名或密码错误');
    }

    // 6. 双因素认证检查
    if (user.twoFactorEnabled) {
      if (!twoFactorCode) {
        throw new BadRequestException('需要双因素认证码');
      }
      await this.verifyTwoFactorCode(user, twoFactorCode);
    }

    // 7. 设备信任检查
    const deviceTrusted = await this.checkDeviceTrust(user.id, deviceFingerprint);
    if (!deviceTrusted) {
      // 发送设备验证通知
      await this.notificationService.sendDeviceVerification(user, clientInfo);
    }

    // 8. 生成会话和令牌
    const session = await this.createUserSession(user, clientInfo);
    const tokens = await this.generateTokens(user, session);

    // 9. 更新用户登录信息
    await this.updateUserLoginInfo(user, clientInfo);

    // 10. 记录成功登录
    await this.logSecurityEvent('LOGIN_SUCCESS', { sessionId: session.id }, clientInfo, user.id);

    // 11. 清除失败登录计数
    await this.clearFailedLoginAttempts(user);

    return {
      user: this.sanitizeUser(user),
      session: {
        id: session.id,
        expiresAt: session.expiresAt,
        deviceTrusted
      },
      ...tokens
    };
  }

  /**
   * 生成访问令牌和刷新令牌 - 使用RS256算法
   * @param user 用户信息
   * @param session 会话信息
   * @returns 令牌对象
   */
  private async generateTokens(user: User, session: UserSession): Promise<TokenPair> {
    const now = Math.floor(Date.now() / 1000);
    const accessTokenExpiry = now + (15 * 60); // 15分钟
    const refreshTokenExpiry = now + (7 * 24 * 60 * 60); // 7天

    // 访问令牌载荷
    const accessPayload = {
      sub: user.id,
      username: user.username,
      email: user.email,
      roles: user.roles?.map(r => r.name) || [],
      permissions: await this.getUserPermissions(user),
      sessionId: session.id,
      iat: now,
      exp: accessTokenExpiry,
      iss: this.configService.get('JWT_ISSUER'),
      aud: this.configService.get('JWT_AUDIENCE'),
    };

    // 刷新令牌载荷
    const refreshPayload = {
      sub: user.id,
      sessionId: session.id,
      type: 'refresh',
      iat: now,
      exp: refreshTokenExpiry,
      iss: this.configService.get('JWT_ISSUER'),
      aud: this.configService.get('JWT_AUDIENCE'),
    };

    const accessToken = this.jwtService.sign(accessPayload, {
      algorithm: 'RS256',
      privateKey: this.configService.get('JWT_PRIVATE_KEY'),
    });

    const refreshToken = this.jwtService.sign(refreshPayload, {
      algorithm: 'RS256',
      privateKey: this.configService.get('JWT_PRIVATE_KEY'),
    });

    // 存储刷新令牌哈希到数据库
    const refreshTokenHash = crypto.createHash('sha256').update(refreshToken).digest('hex');
    session.refreshTokenHash = refreshTokenHash;
    await this.sessionRepository.save(session);

    // 将访问令牌存储到Redis (用于快速验证和撤销)
    await this.redisService.setex(
      `access_token:${user.id}:${session.id}`,
      15 * 60, // 15分钟
      JSON.stringify({ tokenHash: crypto.createHash('sha256').update(accessToken).digest('hex') })
    );

    return {
      accessToken,
      refreshToken,
      expiresIn: 15 * 60,
      tokenType: 'Bearer'
    };
  }

  /**
   * 刷新访问令牌
   * @param refreshToken 刷新令牌
   * @param clientInfo 客户端信息
   * @returns 新的令牌对
   */
  async refreshToken(refreshToken: string, clientInfo: ClientInfo): Promise<TokenPair> {
    try {
      // 验证刷新令牌
      const payload = this.jwtService.verify(refreshToken, {
        publicKey: this.configService.get('JWT_PUBLIC_KEY'),
        algorithms: ['RS256'],
      });

      if (payload.type !== 'refresh') {
        throw new UnauthorizedException('无效的刷新令牌');
      }

      // 查找会话
      const session = await this.sessionRepository.findOne({
        where: { id: payload.sessionId, isActive: true },
        relations: ['user', 'user.roles'],
      });

      if (!session || session.expiresAt < new Date()) {
        throw new UnauthorizedException('会话已过期');
      }

      // 验证刷新令牌哈希
      const refreshTokenHash = crypto.createHash('sha256').update(refreshToken).digest('hex');
      if (session.refreshTokenHash !== refreshTokenHash) {
        // 可能的令牌泄露，撤销所有会话
        await this.revokeAllUserSessions(session.user.id, '检测到令牌泄露');
        throw new UnauthorizedException('检测到安全威胁，已撤销所有会话');
      }

      // 更新会话活动时间
      session.lastActivity = new Date();
      await this.sessionRepository.save(session);

      // 生成新的访问令牌
      const newTokens = await this.generateTokens(session.user, session);

      // 记录令牌刷新事件
      await this.logSecurityEvent('TOKEN_REFRESH', { sessionId: session.id }, clientInfo, session.user.id);

      return newTokens;
    } catch (error) {
      await this.logSecurityEvent('TOKEN_REFRESH_FAILED', { error: error.message }, clientInfo);
      throw new UnauthorizedException('刷新令牌无效');
    }
  }

  /**
   * 用户登出
   * @param sessionId 会话ID
   * @param userId 用户ID
   * @param clientInfo 客户端信息
   */
  async logout(sessionId: string, userId: string, clientInfo: ClientInfo): Promise<void> {
    // 撤销会话
    await this.revokeSession(sessionId, '用户主动登出');

    // 从Redis删除访问令牌
    await this.redisService.del(`access_token:${userId}:${sessionId}`);

    // 记录登出事件
    await this.logSecurityEvent('LOGOUT', { sessionId }, clientInfo, userId);

    this.logger.log(`User ${userId} logged out, session ${sessionId} revoked`);
  }

  /**
   * 撤销用户所有会话
   * @param userId 用户ID
   * @param reason 撤销原因
   */
  async revokeAllUserSessions(userId: string, reason: string): Promise<void> {
    // 撤销数据库中的所有会话
    await this.sessionRepository.update(
      { userId, isActive: true },
      { isActive: false, updatedAt: new Date() }
    );

    // 从Redis删除所有访问令牌
    const pattern = `access_token:${userId}:*`;
    const keys = await this.redisService.keys(pattern);
    if (keys.length > 0) {
      await this.redisService.del(...keys);
    }

    this.logger.warn(`All sessions revoked for user ${userId}, reason: ${reason}`);
  }
}```

#### 权限守卫和中间件 (基于RBAC权限模型)

```typescript
// src/auth/guards/jwt-auth.guard.ts
import { Injectable, ExecutionContext, UnauthorizedException, Logger } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { Reflector } from '@nestjs/core';
import { RedisService } from '../../redis/redis.service';
import { SecurityService } from '../../security/security.service';

@Injectable()
export class JwtAuthGuard extends AuthGuard('jwt') {
  private readonly logger = new Logger(JwtAuthGuard.name);

  constructor(
    private reflector: Reflector,
    private redisService: RedisService,
    private securityService: SecurityService,
  ) {
    super();
  }

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const isPublic = this.reflector.getAllAndOverride<boolean>('isPublic', [
      context.getHandler(),
      context.getClass(),
    ]);

    if (isPublic) {
      return true;
    }

    const request = context.switchToHttp().getRequest();
    const token = this.extractTokenFromHeader(request);

    if (!token) {
      throw new UnauthorizedException('访问令牌缺失');
    }

    // 检查令牌是否在黑名单中
    const isBlacklisted = await this.redisService.get(`blacklist:${token}`);
    if (isBlacklisted) {
      throw new UnauthorizedException('令牌已被撤销');
    }

    // 调用父类验证
    const result = await super.canActivate(context) as boolean;

    if (result) {
      // 更新用户活动时间
      const user = request.user;
      await this.securityService.updateUserActivity(user.id, {
        ipAddress: request.ip,
        userAgent: request.get('User-Agent'),
        endpoint: request.route?.path,
        method: request.method,
      });
    }

    return result;
  }

  private extractTokenFromHeader(request: any): string | undefined {
    const [type, token] = request.headers.authorization?.split(' ') ?? [];
    return type === 'Bearer' ? token : undefined;
  }
}

// src/auth/guards/roles.guard.ts
@Injectable()
export class RolesGuard implements CanActivate {
  private readonly logger = new Logger(RolesGuard.name);

  constructor(
    private reflector: Reflector,
    private securityService: SecurityService,
  ) {}

  /**
   * 检查用户是否具有所需角色权限
   * @param context 执行上下文
   * @returns 是否有权限
   */
  async canActivate(context: ExecutionContext): Promise<boolean> {
    const requiredRoles = this.reflector.getAllAndOverride<string[]>('roles', [
      context.getHandler(),
      context.getClass(),
    ]);

    const requiredPermissions = this.reflector.getAllAndOverride<string[]>('permissions', [
      context.getHandler(),
      context.getClass(),
    ]);

    if (!requiredRoles && !requiredPermissions) {
      return true;
    }

    const request = context.switchToHttp().getRequest();
    const user = request.user;

    if (!user) {
      throw new UnauthorizedException('用户信息缺失');
    }

    // 检查角色权限
    if (requiredRoles) {
      const hasRole = requiredRoles.some(role => 
        user.roles?.some(userRole => userRole.name === role)
      );
      if (!hasRole) {
        this.logger.warn(`User ${user.id} lacks required roles: ${requiredRoles.join(', ')}`);
        return false;
      }
    }

    // 检查细粒度权限
    if (requiredPermissions) {
      const hasPermission = await this.securityService.checkUserPermissions(
        user.id,
        requiredPermissions
      );
      if (!hasPermission) {
        this.logger.warn(`User ${user.id} lacks required permissions: ${requiredPermissions.join(', ')}`);
        return false;
      }
    }

    return true;
  }
}

// src/auth/guards/throttle.guard.ts
@Injectable()
export class ThrottleGuard implements CanActivate {
  private readonly logger = new Logger(ThrottleGuard.name);

  constructor(
    private reflector: Reflector,
    private rateLimiterService: RateLimiterService,
  ) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const throttleOptions = this.reflector.getAllAndOverride<ThrottleOptions>('throttle', [
      context.getHandler(),
      context.getClass(),
    ]);

    if (!throttleOptions) {
      return true;
    }

    const request = context.switchToHttp().getRequest();
    const key = this.generateThrottleKey(request, throttleOptions);

    try {
      await this.rateLimiterService.checkLimit(
        key,
        throttleOptions.limit,
        throttleOptions.ttl
      );
      return true;
    } catch (error) {
      this.logger.warn(`Rate limit exceeded for key: ${key}`);
      throw new UnauthorizedException('请求过于频繁，请稍后再试');
    }
  }

  private generateThrottleKey(request: any, options: ThrottleOptions): string {
    const { keyGenerator = 'ip' } = options;
    
    switch (keyGenerator) {
      case 'ip':
        return `throttle:${request.ip}`;
      case 'user':
        return `throttle:user:${request.user?.id || 'anonymous'}`;
      case 'endpoint':
        return `throttle:endpoint:${request.route?.path}:${request.ip}`;
      default:
        return `throttle:${request.ip}`;
    }
  }
}```
  }
}
```

### API设计 (基于OpenAPI 3.0规范)

#### 认证控制器实现
```typescript
// src/auth/auth.controller.ts
import {
  Controller,
  Post,
  Body,
  UseGuards,
  Request,
  HttpCode,
  HttpStatus,
  Get,
  Patch,
  ValidationPipe,
  UsePipes,
  Ip,
  Headers,
  Logger,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth, ApiBody } from '@nestjs/swagger';
import { ThrottlerGuard } from '@nestjs/throttler';
import { AuthService } from './auth.service';
import { JwtAuthGuard } from './guards/jwt-auth.guard';
import { RolesGuard } from './guards/roles.guard';
import { Public } from './decorators/public.decorator';
import { Roles } from './decorators/roles.decorator';
import { Throttle } from './decorators/throttle.decorator';
import {
  RegisterDto,
  LoginDto,
  RefreshTokenDto,
  ChangePasswordDto,
  ForgotPasswordDto,
  ResetPasswordDto,
  Enable2FADto,
  Verify2FADto,
} from './dto';
import { AuthResult, ClientInfo } from './interfaces';

@ApiTags('认证管理')
@Controller('api/v1/auth')
@UseGuards(ThrottlerGuard)
export class AuthController {
  private readonly logger = new Logger(AuthController.name);

  constructor(private readonly authService: AuthService) {}

  /**
   * 用户注册
   */
  @Public()
  @Post('register')
  @HttpCode(HttpStatus.CREATED)
  @Throttle(3, 300) // 3次/5分钟
  @ApiOperation({ summary: '用户注册', description: '创建新用户账户' })
  @ApiBody({ type: RegisterDto })
  @ApiResponse({
    status: 201,
    description: '注册成功',
    schema: {
      type: 'object',
      properties: {
        success: { type: 'boolean', example: true },
        data: {
          type: 'object',
          properties: {
            user: {
              type: 'object',
              properties: {
                id: { type: 'string', format: 'uuid' },
                username: { type: 'string' },
                email: { type: 'string', format: 'email' },
                displayName: { type: 'string' },
                emailVerified: { type: 'boolean' },
                createdAt: { type: 'string', format: 'date-time' },
              },
            },
            accessToken: { type: 'string' },
            refreshToken: { type: 'string' },
            expiresIn: { type: 'number', example: 900 },
            tokenType: { type: 'string', example: 'Bearer' },
          },
        },
        message: { type: 'string', example: '注册成功' },
      },
    },
  })
  @ApiResponse({ status: 400, description: '请求参数错误' })
  @ApiResponse({ status: 409, description: '用户名或邮箱已存在' })
  @ApiResponse({ status: 429, description: '请求过于频繁' })
  @UsePipes(new ValidationPipe({ transform: true, whitelist: true }))
  async register(
    @Body() registerDto: RegisterDto,
    @Ip() ipAddress: string,
    @Headers('user-agent') userAgent: string,
  ): Promise<{ success: boolean; data: AuthResult; message: string }> {
    const clientInfo: ClientInfo = {
      ipAddress,
      userAgent,
      deviceFingerprint: registerDto.deviceFingerprint,
    };

    const result = await this.authService.register(registerDto, clientInfo);
    
    this.logger.log(`User registered successfully: ${result.user.username}`);
    
    return {
      success: true,
      data: result,
      message: '注册成功',
    };
  }

  /**
   * 用户登录
   */
  @Public()
  @Post('login')
  @HttpCode(HttpStatus.OK)
  @Throttle(5, 300) // 5次/5分钟
  @ApiOperation({ summary: '用户登录', description: '用户身份验证' })
  @ApiBody({ type: LoginDto })
  @ApiResponse({
    status: 200,
    description: '登录成功',
    schema: {
      type: 'object',
      properties: {
        success: { type: 'boolean', example: true },
        data: {
          type: 'object',
          properties: {
            user: {
              type: 'object',
              properties: {
                id: { type: 'string', format: 'uuid' },
                username: { type: 'string' },
                email: { type: 'string', format: 'email' },
                displayName: { type: 'string' },
                roles: {
                  type: 'array',
                  items: { type: 'string' },
                },
                permissions: {
                  type: 'array',
                  items: { type: 'string' },
                },
                lastLoginAt: { type: 'string', format: 'date-time' },
              },
            },
            session: {
              type: 'object',
              properties: {
                id: { type: 'string', format: 'uuid' },
                expiresAt: { type: 'string', format: 'date-time' },
                deviceTrusted: { type: 'boolean' },
              },
            },
            accessToken: { type: 'string' },
            refreshToken: { type: 'string' },
            expiresIn: { type: 'number', example: 900 },
            tokenType: { type: 'string', example: 'Bearer' },
          },
        },
        message: { type: 'string', example: '登录成功' },
      },
    },
  })
  @ApiResponse({ status: 400, description: '请求参数错误' })
  @ApiResponse({ status: 401, description: '用户名或密码错误' })
  @ApiResponse({ status: 423, description: '账户已被锁定' })
  @ApiResponse({ status: 429, description: '请求过于频繁' })
  @UsePipes(new ValidationPipe({ transform: true, whitelist: true }))
  async login(
    @Body() loginDto: LoginDto,
    @Ip() ipAddress: string,
    @Headers('user-agent') userAgent: string,
  ): Promise<{ success: boolean; data: AuthResult; message: string }> {
    const clientInfo: ClientInfo = {
      ipAddress,
      userAgent,
      deviceFingerprint: loginDto.deviceFingerprint,
    };

    const result = await this.authService.login(loginDto, clientInfo);
    
    this.logger.log(`User logged in successfully: ${result.user.username}`);
    
    return {
      success: true,
      data: result,
      message: '登录成功',
    };
  }

  /**
   * 刷新访问令牌
   */
  @Public()
  @Post('refresh')
  @HttpCode(HttpStatus.OK)
  @Throttle(10, 60) // 10次/分钟
  @ApiOperation({ summary: '刷新令牌', description: '使用刷新令牌获取新的访问令牌' })
  @ApiBody({ type: RefreshTokenDto })
  @ApiResponse({
    status: 200,
    description: '令牌刷新成功',
    schema: {
      type: 'object',
      properties: {
        success: { type: 'boolean', example: true },
        data: {
          type: 'object',
          properties: {
            accessToken: { type: 'string' },
            refreshToken: { type: 'string' },
            expiresIn: { type: 'number', example: 900 },
            tokenType: { type: 'string', example: 'Bearer' },
          },
        },
        message: { type: 'string', example: '令牌刷新成功' },
      },
    },
  })
  @ApiResponse({ status: 401, description: '刷新令牌无效或已过期' })
  @UsePipes(new ValidationPipe({ transform: true, whitelist: true }))
  async refreshToken(
    @Body() refreshTokenDto: RefreshTokenDto,
    @Ip() ipAddress: string,
    @Headers('user-agent') userAgent: string,
  ): Promise<{ success: boolean; data: any; message: string }> {
    const clientInfo: ClientInfo = {
      ipAddress,
      userAgent,
      deviceFingerprint: refreshTokenDto.deviceFingerprint,
    };

    const tokens = await this.authService.refreshToken(
      refreshTokenDto.refreshToken,
      clientInfo,
    );
    
    return {
      success: true,
      data: tokens,
      message: '令牌刷新成功',
    };
  }

  /**
   * 用户登出
   */
  @Post('logout')
  @HttpCode(HttpStatus.OK)
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: '用户登出', description: '撤销当前会话' })
  @ApiResponse({
    status: 200,
    description: '登出成功',
    schema: {
      type: 'object',
      properties: {
        success: { type: 'boolean', example: true },
        message: { type: 'string', example: '登出成功' },
      },
    },
  })
  @ApiResponse({ status: 401, description: '未授权访问' })
  async logout(
    @Request() req: any,
    @Ip() ipAddress: string,
    @Headers('user-agent') userAgent: string,
  ): Promise<{ success: boolean; message: string }> {
    const clientInfo: ClientInfo = {
      ipAddress,
      userAgent,
    };

    await this.authService.logout(req.user.sessionId, req.user.sub, clientInfo);
    
    return {
      success: true,
      message: '登出成功',
    };
  }

  /**
   * 获取当前用户信息
   */
  @Get('me')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: '获取当前用户信息', description: '获取已认证用户的详细信息' })
  @ApiResponse({
    status: 200,
    description: '获取成功',
    schema: {
      type: 'object',
      properties: {
        success: { type: 'boolean', example: true },
        data: {
          type: 'object',
          properties: {
            id: { type: 'string', format: 'uuid' },
            username: { type: 'string' },
            email: { type: 'string', format: 'email' },
            displayName: { type: 'string' },
            avatar: { type: 'string', format: 'uri' },
            roles: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  id: { type: 'string', format: 'uuid' },
                  name: { type: 'string' },
                  displayName: { type: 'string' },
                  permissions: {
                    type: 'array',
                    items: { type: 'string' },
                  },
                },
              },
            },
            emailVerified: { type: 'boolean' },
            twoFactorEnabled: { type: 'boolean' },
            lastLoginAt: { type: 'string', format: 'date-time' },
            createdAt: { type: 'string', format: 'date-time' },
            updatedAt: { type: 'string', format: 'date-time' },
          },
        },
      },
    },
  })
  @ApiResponse({ status: 401, description: '未授权访问' })
  async getCurrentUser(@Request() req: any): Promise<{ success: boolean; data: any }> {
    const user = await this.authService.getCurrentUser(req.user.sub);
    
    return {
      success: true,
      data: user,
    };
  }

  /**
   * 修改密码
   */
  @Patch('password')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @Throttle(3, 300) // 3次/5分钟
  @ApiOperation({ summary: '修改密码', description: '用户修改登录密码' })
  @ApiBody({ type: ChangePasswordDto })
  @ApiResponse({
    status: 200,
    description: '密码修改成功',
    schema: {
      type: 'object',
      properties: {
        success: { type: 'boolean', example: true },
        message: { type: 'string', example: '密码修改成功' },
      },
    },
  })
  @ApiResponse({ status: 400, description: '原密码错误' })
  @ApiResponse({ status: 401, description: '未授权访问' })
  @UsePipes(new ValidationPipe({ transform: true, whitelist: true }))
  async changePassword(
    @Request() req: any,
    @Body() changePasswordDto: ChangePasswordDto,
    @Ip() ipAddress: string,
    @Headers('user-agent') userAgent: string,
  ): Promise<{ success: boolean; message: string }> {
    const clientInfo: ClientInfo = {
      ipAddress,
      userAgent,
    };

    await this.authService.changePassword(
      req.user.sub,
      changePasswordDto,
      clientInfo,
    );
    
    return {
      success: true,
      message: '密码修改成功',
    };
  }

  /**
   * 启用双因素认证
   */
  @Post('2fa/enable')
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: '启用双因素认证', description: '为用户账户启用2FA' })
  @ApiResponse({
    status: 200,
    description: '2FA启用成功',
    schema: {
      type: 'object',
      properties: {
        success: { type: 'boolean', example: true },
        data: {
          type: 'object',
          properties: {
            qrCode: { type: 'string', description: 'Base64编码的二维码图片' },
            secret: { type: 'string', description: '备用密钥' },
            backupCodes: {
              type: 'array',
              items: { type: 'string' },
              description: '备用恢复码',
            },
          },
        },
        message: { type: 'string', example: '请扫描二维码完成设置' },
      },
    },
  })
  @ApiResponse({ status: 401, description: '未授权访问' })
  async enable2FA(
    @Request() req: any,
  ): Promise<{ success: boolean; data: any; message: string }> {
    const result = await this.authService.enable2FA(req.user.sub);
    
    return {
      success: true,
      data: result,
      message: '请扫描二维码完成设置',
    };
  }
}
```

#### 数据传输对象 (DTO) 定义
```typescript
// src/auth/dto/register.dto.ts
import {
  IsEmail,
  IsString,
  IsOptional,
  MinLength,
  MaxLength,
  Matches,
  IsNotEmpty,
} from 'class-validator';
import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { Transform } from 'class-transformer';

export class RegisterDto {
  @ApiProperty({
    description: '用户名',
    example: 'john_doe',
    minLength: 3,
    maxLength: 50,
  })
  @IsString()
  @IsNotEmpty({ message: '用户名不能为空' })
  @MinLength(3, { message: '用户名至少3个字符' })
  @MaxLength(50, { message: '用户名最多50个字符' })
  @Matches(/^[a-zA-Z0-9_]+$/, { message: '用户名只能包含字母、数字和下划线' })
  @Transform(({ value }) => value?.toLowerCase())
  username: string;

  @ApiProperty({
    description: '邮箱地址',
    example: 'john@example.com',
    format: 'email',
  })
  @IsEmail({}, { message: '邮箱格式不正确' })
  @IsNotEmpty({ message: '邮箱不能为空' })
  @Transform(({ value }) => value?.toLowerCase())
  email: string;

  @ApiProperty({
    description: '密码',
    example: 'SecurePass123!',
    minLength: 8,
    maxLength: 128,
  })
  @IsString()
  @IsNotEmpty({ message: '密码不能为空' })
  @MinLength(8, { message: '密码至少8个字符' })
  @MaxLength(128, { message: '密码最多128个字符' })
  @Matches(
    /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/,
    { message: '密码必须包含大小写字母、数字和特殊字符' },
  )
  password: string;

  @ApiPropertyOptional({
    description: '显示名称',
    example: 'John Doe',
    maxLength: 100,
  })
  @IsOptional()
  @IsString()
  @MaxLength(100, { message: '显示名称最多100个字符' })
  displayName?: string;

  @ApiPropertyOptional({
    description: '设备指纹',
    example: 'fp_1234567890abcdef',
  })
  @IsOptional()
  @IsString()
  deviceFingerprint?: string;

  @ApiPropertyOptional({
    description: '邀请码',
    example: 'INV123456',
  })
  @IsOptional()
  @IsString()
  inviteCode?: string;
}

// src/auth/dto/login.dto.ts
export class LoginDto {
  @ApiProperty({
    description: '用户名或邮箱',
    example: 'john_doe',
  })
  @IsString()
  @IsNotEmpty({ message: '用户名不能为空' })
  username: string;

  @ApiProperty({
    description: '密码',
    example: 'SecurePass123!',
  })
  @IsString()
  @IsNotEmpty({ message: '密码不能为空' })
  password: string;

  @ApiPropertyOptional({
    description: '双因素认证码',
    example: '123456',
  })
  @IsOptional()
  @IsString()
  @Matches(/^\d{6}$/, { message: '双因素认证码必须是6位数字' })
  twoFactorCode?: string;

  @ApiPropertyOptional({
    description: '验证码',
    example: 'ABCD',
  })
  @IsOptional()
  @IsString()
  captcha?: string;

  @ApiPropertyOptional({
    description: '设备指纹',
    example: 'fp_1234567890abcdef',
  })
  @IsOptional()
  @IsString()
  deviceFingerprint?: string;

  @ApiPropertyOptional({
    description: '记住登录状态',
    example: true,
  })
  @IsOptional()
  @Transform(({ value }) => value === 'true' || value === true)
  rememberMe?: boolean;
}

// src/auth/dto/refresh-token.dto.ts
export class RefreshTokenDto {
  @ApiProperty({
    description: '刷新令牌',
    example: 'eyJhbGciOiJSUzI1NiIs...',
  })
  @IsString()
  @IsNotEmpty({ message: '刷新令牌不能为空' })
  refreshToken: string;

  @ApiPropertyOptional({
    description: '设备指纹',
    example: 'fp_1234567890abcdef',
  })
  @IsOptional()
  @IsString()
  deviceFingerprint?: string;
}

// src/auth/dto/change-password.dto.ts
export class ChangePasswordDto {
  @ApiProperty({
    description: '当前密码',
    example: 'OldPass123!',
  })
  @IsString()
  @IsNotEmpty({ message: '当前密码不能为空' })
  currentPassword: string;

  @ApiProperty({
    description: '新密码',
    example: 'NewPass123!',
    minLength: 8,
    maxLength: 128,
  })
  @IsString()
  @IsNotEmpty({ message: '新密码不能为空' })
  @MinLength(8, { message: '新密码至少8个字符' })
  @MaxLength(128, { message: '新密码最多128个字符' })
  @Matches(
    /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/,
    { message: '新密码必须包含大小写字母、数字和特殊字符' },
  )
  newPassword: string;
}
```

#### 错误处理和响应格式
```typescript
// src/common/filters/http-exception.filter.ts
import {
  ExceptionFilter,
  Catch,
  ArgumentsHost,
  HttpException,
  HttpStatus,
  Logger,
} from '@nestjs/common';
import { Request, Response } from 'express';
import { SecurityService } from '../security/security.service';

@Catch()
export class HttpExceptionFilter implements ExceptionFilter {
  private readonly logger = new Logger(HttpExceptionFilter.name);

  constructor(private readonly securityService: SecurityService) {}

  /**
   * 捕获并处理所有HTTP异常
   * @param exception 异常对象
   * @param host 参数主机
   */
  catch(exception: unknown, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();
    const request = ctx.getRequest<Request>();

    let status = HttpStatus.INTERNAL_SERVER_ERROR;
    let message = '服务器内部错误';
    let code = 'INTERNAL_SERVER_ERROR';
    let details: any = null;

    if (exception instanceof HttpException) {
      status = exception.getStatus();
      const exceptionResponse = exception.getResponse();
      
      if (typeof exceptionResponse === 'object') {
        message = (exceptionResponse as any).message || exception.message;
        code = (exceptionResponse as any).code || this.getErrorCode(status);
        details = (exceptionResponse as any).details;
      } else {
        message = exceptionResponse as string;
        code = this.getErrorCode(status);
      }
    } else if (exception instanceof Error) {
      message = exception.message;
      this.logger.error(`Unexpected error: ${exception.message}`, exception.stack);
    }

    // 记录安全相关错误
    if (this.isSecurityRelatedError(status)) {
      this.securityService.logSecurityEvent('SECURITY_ERROR', {
        status,
        message,
        path: request.url,
        method: request.method,
        userAgent: request.get('User-Agent'),
        ipAddress: request.ip,
      });
    }

    const errorResponse = {
      success: false,
      error: {
        code,
        message: Array.isArray(message) ? message : [message],
        timestamp: new Date().toISOString(),
        path: request.url,
        method: request.method,
        ...(details && { details }),
        ...(process.env.NODE_ENV === 'development' && {
          stack: exception instanceof Error ? exception.stack : undefined,
        }),
      },
      requestId: request.headers['x-request-id'] || 'unknown',
    };

    this.logger.error(
      `HTTP ${status} Error: ${JSON.stringify(errorResponse)}`,
      exception instanceof Error ? exception.stack : undefined,
    );

    response.status(status).json(errorResponse);
  }

  /**
   * 根据HTTP状态码获取错误代码
   * @param status HTTP状态码
   * @returns 错误代码
   */
  private getErrorCode(status: number): string {
    const errorCodes = {
      400: 'BAD_REQUEST',
      401: 'UNAUTHORIZED',
      403: 'FORBIDDEN',
      404: 'NOT_FOUND',
      409: 'CONFLICT',
      422: 'UNPROCESSABLE_ENTITY',
      423: 'LOCKED',
      429: 'TOO_MANY_REQUESTS',
      500: 'INTERNAL_SERVER_ERROR',
      502: 'BAD_GATEWAY',
      503: 'SERVICE_UNAVAILABLE',
    };
    return errorCodes[status] || 'UNKNOWN_ERROR';
  }

  /**
   * 判断是否为安全相关错误
   * @param status HTTP状态码
   * @returns 是否为安全相关错误
   */
  private isSecurityRelatedError(status: number): boolean {
    return [401, 403, 423, 429].includes(status);
  }
}

// src/auth/interfaces/auth.interface.ts
export interface AuthResult {
  user: {
    id: string;
    username: string;
    email: string;
    displayName?: string;
    avatar?: string;
    roles: string[];
    permissions: string[];
    emailVerified: boolean;
    twoFactorEnabled: boolean;
    lastLoginAt?: Date;
    createdAt: Date;
  };
  session?: {
    id: string;
    expiresAt: Date;
    deviceTrusted: boolean;
  };
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
  tokenType: string;
}

export interface TokenPair {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
  tokenType: string;
}

export interface ClientInfo {
  ipAddress: string;
  userAgent: string;
  deviceFingerprint?: string;
}

export interface SecurityEvent {
  type: string;
  userId?: string;
  data: Record<string, any>;
  clientInfo: ClientInfo;
  timestamp: Date;
  riskScore?: number;
}

// src/auth/decorators/public.decorator.ts
import { SetMetadata } from '@nestjs/common';

export const IS_PUBLIC_KEY = 'isPublic';
export const Public = () => SetMetadata(IS_PUBLIC_KEY, true);

// src/auth/decorators/roles.decorator.ts
import { SetMetadata } from '@nestjs/common';

export const ROLES_KEY = 'roles';
export const Roles = (...roles: string[]) => SetMetadata(ROLES_KEY, roles);

// src/auth/decorators/permissions.decorator.ts
import { SetMetadata } from '@nestjs/common';

export const PERMISSIONS_KEY = 'permissions';
export const Permissions = (...permissions: string[]) => 
  SetMetadata(PERMISSIONS_KEY, permissions);

// src/auth/decorators/throttle.decorator.ts
import { SetMetadata } from '@nestjs/common';

export interface ThrottleOptions {
  limit: number;
  ttl: number;
  keyGenerator?: 'ip' | 'user' | 'endpoint';
}

export const THROTTLE_KEY = 'throttle';
export const Throttle = (limit: number, ttl: number, keyGenerator: 'ip' | 'user' | 'endpoint' = 'ip') => 
  SetMetadata(THROTTLE_KEY, { limit, ttl, keyGenerator });
```

#### 认证模块配置
```typescript
// src/auth/auth.module.ts
import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { JwtModule } from '@nestjs/jwt';
import { PassportModule } from '@nestjs/passport';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { ThrottlerModule } from '@nestjs/throttler';
import { AuthController } from './auth.controller';
import { AuthService } from './auth.service';
import { SecurityService } from './security.service';
import { RateLimiterService } from './rate-limiter.service';
import { NotificationService } from './notification.service';
import { JwtStrategy } from './strategies/jwt.strategy';
import { JwtAuthGuard } from './guards/jwt-auth.guard';
import { RolesGuard } from './guards/roles.guard';
import { ThrottleGuard } from './guards/throttle.guard';
import { User, Role, UserRole, UserSession, SecurityLog } from './entities';
import { RedisModule } from '../redis/redis.module';

@Module({
  imports: [
    TypeOrmModule.forFeature([
      User,
      Role,
      UserRole,
      UserSession,
      SecurityLog,
    ]),
    PassportModule.register({ defaultStrategy: 'jwt' }),
    JwtModule.registerAsync({
      imports: [ConfigModule],
      useFactory: async (configService: ConfigService) => ({
        privateKey: configService.get<string>('JWT_PRIVATE_KEY'),
        publicKey: configService.get<string>('JWT_PUBLIC_KEY'),
        signOptions: {
          algorithm: 'RS256',
          issuer: configService.get<string>('JWT_ISSUER'),
          audience: configService.get<string>('JWT_AUDIENCE'),
        },
        verifyOptions: {
          algorithms: ['RS256'],
          issuer: configService.get<string>('JWT_ISSUER'),
          audience: configService.get<string>('JWT_AUDIENCE'),
        },
      }),
      inject: [ConfigService],
    }),
    ThrottlerModule.forRootAsync({
      imports: [ConfigModule],
      useFactory: (configService: ConfigService) => ({
        ttl: configService.get<number>('THROTTLE_TTL', 60),
        limit: configService.get<number>('THROTTLE_LIMIT', 10),
        storage: configService.get<string>('REDIS_URL') ? 'redis' : 'memory',
      }),
      inject: [ConfigService],
    }),
    RedisModule,
    ConfigModule,
  ],
  controllers: [AuthController],
  providers: [
    AuthService,
    SecurityService,
    RateLimiterService,
    NotificationService,
    JwtStrategy,
    JwtAuthGuard,
    RolesGuard,
    ThrottleGuard,
  ],
  exports: [
    AuthService,
    SecurityService,
    JwtAuthGuard,
    RolesGuard,
    ThrottleGuard,
  ],
})
export class AuthModule {}

// src/auth/strategies/jwt.strategy.ts
import { Injectable, UnauthorizedException } from '@nestjs/common';
import { PassportStrategy } from '@nestjs/passport';
import { ExtractJwt, Strategy } from 'passport-jwt';
import { ConfigService } from '@nestjs/config';
import { AuthService } from '../auth.service';

@Injectable()
export class JwtStrategy extends PassportStrategy(Strategy) {
  constructor(
    private configService: ConfigService,
    private authService: AuthService,
  ) {
    super({
      jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
      ignoreExpiration: false,
      secretOrKey: configService.get<string>('JWT_PUBLIC_KEY'),
      algorithms: ['RS256'],
      issuer: configService.get<string>('JWT_ISSUER'),
      audience: configService.get<string>('JWT_AUDIENCE'),
    });
  }

  /**
   * JWT策略验证方法
   * @param payload JWT载荷
   * @returns 用户信息
   */
  async validate(payload: any) {
    const { sub: userId, sessionId, exp } = payload;

    // 检查令牌是否过期
    if (Date.now() >= exp * 1000) {
      throw new UnauthorizedException('访问令牌已过期');
    }

    // 验证会话是否有效
    const isSessionValid = await this.authService.validateSession(sessionId, userId);
    if (!isSessionValid) {
      throw new UnauthorizedException('会话无效或已过期');
    }

    // 返回用户信息（将被注入到request.user中）
    return {
      sub: userId,
      username: payload.username,
      email: payload.email,
      roles: payload.roles || [],
      permissions: payload.permissions || [],
      sessionId,
    };
  }
}

// src/config/auth.config.ts
import { registerAs } from '@nestjs/config';

export default registerAs('auth', () => ({
  // JWT配置
  jwt: {
    privateKey: process.env.JWT_PRIVATE_KEY?.replace(/\\n/g, '\n'),
    publicKey: process.env.JWT_PUBLIC_KEY?.replace(/\\n/g, '\n'),
    issuer: process.env.JWT_ISSUER || 'historical-text-platform',
    audience: process.env.JWT_AUDIENCE || 'historical-text-users',
    accessTokenExpiry: parseInt(process.env.JWT_ACCESS_TOKEN_EXPIRY || '900'), // 15分钟
    refreshTokenExpiry: parseInt(process.env.JWT_REFRESH_TOKEN_EXPIRY || '604800'), // 7天
  },

  // 密码策略
  password: {
    saltRounds: parseInt(process.env.PASSWORD_SALT_ROUNDS || '12'),
    minLength: parseInt(process.env.PASSWORD_MIN_LENGTH || '8'),
    maxLength: parseInt(process.env.PASSWORD_MAX_LENGTH || '128'),
    requireUppercase: process.env.PASSWORD_REQUIRE_UPPERCASE === 'true',
    requireLowercase: process.env.PASSWORD_REQUIRE_LOWERCASE === 'true',
    requireNumbers: process.env.PASSWORD_REQUIRE_NUMBERS === 'true',
    requireSymbols: process.env.PASSWORD_REQUIRE_SYMBOLS === 'true',
  },

  // 账户安全
  security: {
    maxFailedAttempts: parseInt(process.env.MAX_FAILED_ATTEMPTS || '5'),
    lockoutDuration: parseInt(process.env.LOCKOUT_DURATION || '900000'), // 15分钟
    sessionTimeout: parseInt(process.env.SESSION_TIMEOUT || '86400000'), // 24小时
    maxConcurrentSessions: parseInt(process.env.MAX_CONCURRENT_SESSIONS || '5'),
    requireEmailVerification: process.env.REQUIRE_EMAIL_VERIFICATION === 'true',
    enable2FA: process.env.ENABLE_2FA === 'true',
  },

  // 速率限制
  rateLimit: {
    login: {
      windowMs: parseInt(process.env.LOGIN_RATE_WINDOW || '300000'), // 5分钟
      max: parseInt(process.env.LOGIN_RATE_MAX || '5'),
    },
    register: {
      windowMs: parseInt(process.env.REGISTER_RATE_WINDOW || '3600000'), // 1小时
      max: parseInt(process.env.REGISTER_RATE_MAX || '3'),
    },
    passwordReset: {
      windowMs: parseInt(process.env.PASSWORD_RESET_RATE_WINDOW || '3600000'), // 1小时
      max: parseInt(process.env.PASSWORD_RESET_RATE_MAX || '3'),
    },
  },

  // 双因素认证
  twoFactor: {
    issuer: process.env.TWO_FACTOR_ISSUER || 'Historical Text Platform',
    window: parseInt(process.env.TWO_FACTOR_WINDOW || '1'),
    backupCodesCount: parseInt(process.env.BACKUP_CODES_COUNT || '10'),
  },

  // 邮件配置
  email: {
    from: process.env.EMAIL_FROM || 'noreply@historical-text.com',
    verificationExpiry: parseInt(process.env.EMAIL_VERIFICATION_EXPIRY || '86400000'), // 24小时
    passwordResetExpiry: parseInt(process.env.PASSWORD_RESET_EXPIRY || '3600000'), // 1小时
  },
}));
```

### 前端集成

#### Vue3 认证状态管理
```typescript
// stores/auth.ts (Pinia)
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { User, LoginCredentials, AuthResult } from '@/types/auth'
import { authApi } from '@/api/auth'
import { ElMessage } from 'element-plus'

export const useAuthStore = defineStore('auth', () => {
  const user = ref<User | null>(null)
  const accessToken = ref<string | null>(localStorage.getItem('accessToken'))
  const refreshToken = ref<string | null>(localStorage.getItem('refreshToken'))
  const sessionInfo = ref<any>(null)
  const loading = ref(false)

  const isAuthenticated = computed(() => !!accessToken.value && !!user.value)
  const hasRole = computed(() => (role: string) => {
    return user.value?.roles?.includes(role) ?? false
  })
  const hasPermission = computed(() => (permission: string) => {
    return user.value?.permissions?.includes(permission) ?? false
  })

  /**
   * 用户登录
   * @param credentials 登录凭据
   */
  async function login(credentials: LoginCredentials): Promise<AuthResult> {
    loading.value = true
    try {
      const response = await authApi.login(credentials)
      
      if (response.success) {
        user.value = response.data.user
        accessToken.value = response.data.accessToken
        refreshToken.value = response.data.refreshToken
        sessionInfo.value = response.data.session
        
        // 存储到本地存储
        localStorage.setItem('accessToken', response.data.accessToken)
        localStorage.setItem('refreshToken', response.data.refreshToken)
        localStorage.setItem('user', JSON.stringify(response.data.user))
        
        ElMessage.success(response.message || '登录成功')
        return response
      } else {
        throw new Error(response.message || '登录失败')
      }
    } catch (error: any) {
      ElMessage.error(error.message || '登录失败')
      throw error
    } finally {
      loading.value = false
    }
  }

  /**
   * 用户注册
   * @param registerData 注册数据
   */
  async function register(registerData: any): Promise<AuthResult> {
    loading.value = true
    try {
      const response = await authApi.register(registerData)
      
      if (response.success) {
        user.value = response.data.user
        accessToken.value = response.data.accessToken
        refreshToken.value = response.data.refreshToken
        
        localStorage.setItem('accessToken', response.data.accessToken)
        localStorage.setItem('refreshToken', response.data.refreshToken)
        localStorage.setItem('user', JSON.stringify(response.data.user))
        
        ElMessage.success(response.message || '注册成功')
        return response
      } else {
        throw new Error(response.message || '注册失败')
      }
    } catch (error: any) {
      ElMessage.error(error.message || '注册失败')
      throw error
    } finally {
      loading.value = false
    }
  }

  /**
   * 刷新访问令牌
   */
  async function refreshAccessToken(): Promise<boolean> {
    if (!refreshToken.value) {
      await logout()
      return false
    }

    try {
      const response = await authApi.refreshToken({
        refreshToken: refreshToken.value
      })
      
      if (response.success) {
        accessToken.value = response.data.accessToken
        refreshToken.value = response.data.refreshToken
        
        localStorage.setItem('accessToken', response.data.accessToken)
        localStorage.setItem('refreshToken', response.data.refreshToken)
        
        return true
      } else {
        await logout()
        return false
      }
    } catch (error) {
      await logout()
      return false
    }
  }

  /**
   * 用户登出
   */
  async function logout() {
    loading.value = true
    try {
      if (accessToken.value) {
        await authApi.logout()
      }
    } catch (error) {
      console.error('Logout error:', error)
    } finally {
      // 清除状态
      user.value = null
      accessToken.value = null
      refreshToken.value = null
      sessionInfo.value = null
      
      // 清除本地存储
      localStorage.removeItem('accessToken')
      localStorage.removeItem('refreshToken')
      localStorage.removeItem('user')
      
      loading.value = false
      ElMessage.success('已安全登出')
    }
  }

  /**
   * 获取当前用户信息
   */
  async function getCurrentUser() {
    if (!accessToken.value) return null
    
    try {
      const response = await authApi.getCurrentUser()
      if (response.success) {
        user.value = response.data
        localStorage.setItem('user', JSON.stringify(response.data))
        return response.data
      }
    } catch (error) {
      console.error('Get current user error:', error)
      await logout()
    }
    return null
  }

  /**
   * 初始化认证状态
   */
  function initializeAuth() {
    const storedUser = localStorage.getItem('user')
    if (storedUser && accessToken.value) {
      try {
        user.value = JSON.parse(storedUser)
      } catch (error) {
        console.error('Parse stored user error:', error)
        logout()
      }
    }
  }

  return {
    user,
    accessToken,
    refreshToken,
    sessionInfo,
    loading,
    isAuthenticated,
    hasRole,
    hasPermission,
    login,
    register,
    logout,
    refreshAccessToken,
    getCurrentUser,
    initializeAuth
  }
})
```

#### Vue Router 守卫
```typescript
// router/guards.ts
import type { Router } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { ElMessage } from 'element-plus'

export function setupAuthGuards(router: Router) {
  // 认证守卫
  router.beforeEach(async (to, from, next) => {
    const authStore = useAuthStore()
    
    // 初始化认证状态
    if (authStore.accessToken && !authStore.user) {
      await authStore.getCurrentUser()
    }
    
    // 检查路由是否需要认证
    if (to.meta.requiresAuth && !authStore.isAuthenticated) {
      ElMessage.warning('请先登录')
      next({ 
        name: 'login', 
        query: { redirect: to.fullPath } 
      })
      return
    }
    
    // 检查角色权限
    if (to.meta.roles && authStore.user) {
      const hasRequiredRole = (to.meta.roles as string[]).some(role => 
        authStore.hasRole(role)
      )
      
      if (!hasRequiredRole) {
        ElMessage.error('权限不足')
        next({ name: 'forbidden' })
        return
      }
    }
    
    // 检查细粒度权限
    if (to.meta.permissions && authStore.user) {
      const hasRequiredPermission = (to.meta.permissions as string[]).some(permission => 
        authStore.hasPermission(permission)
      )
      
      if (!hasRequiredPermission) {
        ElMessage.error('权限不足')
        next({ name: 'forbidden' })
        return
      }
    }
    
    // 已登录用户访问登录页面，重定向到首页
    if (to.name === 'login' && authStore.isAuthenticated) {
      next({ name: 'dashboard' })
      return
    }
    
    next()
  })
}
```

## 验收标准

### 功能验收
- [ ] 用户注册功能正常
- [ ] 用户登录/登出功能正常
- [ ] JWT令牌生成和验证正确
- [ ] 刷新令牌机制工作正常
- [ ] 角色权限控制有效
- [ ] 密码加密存储安全
- [ ] API接口文档完整

### 性能验收
- [ ] 登录响应时间 < 500ms
- [ ] 令牌验证时间 < 50ms
- [ ] 并发登录支持 > 100用户
- [ ] 数据库查询优化

### 安全验收
- [ ] 密码强度验证
- [ ] 防止暴力破解攻击
- [ ] JWT令牌安全配置
- [ ] 敏感信息不泄露
- [ ] HTTPS通信加密

## 业务价值
- 提供统一的用户身份认证机制
- 确保系统数据和功能的安全访问
- 支持细粒度的权限控制
- 为其他微服务提供认证基础

## 依赖关系
- **前置条件**: Story 1.1 (微服务基础架构)
- **后续依赖**: Story 1.3, 1.4, Epic 2-4所有服务

## 风险与缓解
- **风险**: JWT令牌泄露风险
- **缓解**: 短期访问令牌 + 刷新令牌机制
- **风险**: 数据库性能瓶颈
- **缓解**: Redis缓存 + 数据库索引优化

## 开发任务分解
1. 数据模型设计和数据库迁移 (1天)
2. 认证服务核心逻辑开发 (2天)
3. API接口开发和测试 (1天)
4. 前端认证状态管理 (1天)
5. 安全测试和文档编写 (1天)