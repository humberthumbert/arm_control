#!/usr/bin/env python
# -*- coding: utf-8 -*-

from OpenGL.GL import *
from OpenGL.GL import shaders as GL_shaders
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
from ctypes import *

# uniform mat4 View;  View * 
VERTEX_SHADER="""
#version 330
layout (location = 0) in vec3 Position;

void main()
{
    gl_Position = vec4(Position.x, Position.y, Position.z, 1.0);
}
"""

FRAGMENT_SHADER="""
#version 330
out vec4 FragColor;
void main()
{
    FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
}
"""

class Shader:
    def __init__(self, vertexPath, fragmentPath, geometryPath=None):
        with open(vertexPath, 'r') as vShaderFile, open(fragmentPath, 'r') as fShaderFile:
            self.vShaderStr = vShaderFile.read()
            self.fShaderStr = fShaderFile.read()
        if geometryPath is not None:
            with open(geometryPath, 'r') as gShaderFile:
                self.gShaderStr = gShaderFile.read()
        
        # Create Shader Program
        self.shader = glCreateProgram()

        # Compile Shader
        vertex = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex, VERTEX_SHADER)
        glCompileShader(vertex)
        self.checkCompileErrors(vertex, "VERTEX")
        fragment = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment, FRAGMENT_SHADER)
        glCompileShader(fragment)
        self.checkCompileErrors(fragment, "FRAGMENT")
        if geometryPath is not None:
            geometry = glCreateShader(GL_GEOMETRY_SHADER)
            glShaderSource(geometry, self.gShaderStr)
            glCompileShader(geometry)
            self.checkCompileErrors(geometry, "GEOMETRY")
        
        # Attach shader
        glAttachShader(self.shader, vertex)
        glAttachShader(self.shader, fragment)
        if geometryPath is not None:
            glAttachShader(self.shader, geometry)
        glLinkProgram(self.shader)
        glUseProgram(self.shader)
        self.checkCompileErrors(self.shader, "PROGRAM")

        # Delete Shader
        glDeleteShader(vertex)
        glDeleteShader(fragment)
        if geometryPath is not None:
            glDeleteShader(geometry)
    

    def use(self):
        """ activate the shader"""
        glUseProgram(self.shader)

    def setMat4(self, name, mat):
        glUniformMatrix4fv(glGetUniformLocation(self.shader, name), 1, GL_FALSE, mat)

    def checkCompileErrors(self, shader, shader_type):
        success = GL_FALSE
        infoLog = ""
        if (shader_type != "PROGRAM"):
            success = glGetShaderiv(self.shader, GL_COMPILE_STATUS)
            if success == GL_FALSE:
                infoLog += glGetShaderInfoLog(self.shader)
                print("ERROR::SHADER_COMPILATION_ERROR of type: {}\n{}\n -- \
                ----------------------------------------".format(shader_type, infoLog))
        else:
            success = glGetProgramiv(self.shader, GL_LINK_STATUS)
            if success == GL_FALSE:
                infoLog += glGetProgramInfoLog(self.shader)
                print("ERROR::PROGRAM_LINKING_ERROR of type: {}\n{}\n -- \
                ----------------------------------------".format(shader_type, infoLog))

if __name__=="__main__":
    folderPath = os.path.dirname(os.path.abspath(__file__))
    shader = Shader(os.path.join(folderPath, "vertex.glsl"), 
                    os.path.join(folderPath, "fragment.glsl"))
    shader.use()